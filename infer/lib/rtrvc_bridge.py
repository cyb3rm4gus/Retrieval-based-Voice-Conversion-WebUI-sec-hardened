"""Bridge between sounddevice audio I/O and rtrvc.RVC real-time inference.

All inference runs on CUDA in a dedicated worker thread.
Audio callbacks only copy data (sub-millisecond) to avoid PortAudio xruns.
"""

import atexit
import collections
import logging
import multiprocessing
import threading
import time

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

from configs.config import Config

logger = logging.getLogger(__name__)


class RealtimeVCBridge:

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.rvc = None
        self._running = False
        self._last_rvc = None
        self._last_infer_ms = 0.0
        # Dummy queues — only used if harvest f0 with n_cpu>1, which we avoid
        self._inp_q = multiprocessing.Queue()
        self._opt_q = multiprocessing.Queue()
        atexit.register(self._cleanup)

    def list_input_devices(self):
        devices = sd.query_devices()
        return [
            (i, d["name"])
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    def list_output_devices(self):
        devices = sd.query_devices()
        return [
            (i, d["name"])
            for i, d in enumerate(devices)
            if d["max_output_channels"] > 0
        ]

    def start(
        self,
        pth_path,
        index_path,
        input_device_idx,
        output_device_idx,
        pitch=0,
        f0method="rmvpe",
        index_rate=0.0,
        block_time=0.25,
        crossfade_time=0.05,
        extra_time=2.5,
    ):
        if self._running:
            self.stop()

        from infer.lib import rtrvc as rvc_for_realtime

        torch.cuda.empty_cache()
        self.rvc = rvc_for_realtime.RVC(
            pitch,
            0,  # formant shift
            pth_path,
            index_path,
            index_rate,
            1,  # n_cpu=1 to avoid multiprocessing for harvest fallback
            self._inp_q,
            self._opt_q,
            self.config,
            self._last_rvc,
        )
        self._last_rvc = self.rvc
        self.f0method = f0method

        # Use input device's default sample rate (mic may not support model's tgt_sr)
        self.samplerate = int(sd.query_devices(input_device_idx)["default_samplerate"])
        self.channels = self._get_device_channels(output_device_idx)

        # Frame calculations aligned to zc (10ms boundary)
        self.zc = self.samplerate // 100
        self.block_frame = (
            int(np.round(block_time * self.samplerate / self.zc)) * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(crossfade_time * self.samplerate / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(extra_time * self.samplerate / self.zc)) * self.zc
        )

        # Allocate buffers on CUDA
        total_input = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
        )
        self.input_wav = torch.zeros(
            total_input, device=self.device, dtype=torch.float32
        )
        self.input_wav_res = torch.zeros(
            160 * total_input // self.zc, device=self.device, dtype=torch.float32
        )
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.device, dtype=torch.float32
        )
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc

        # Fade windows for SOLA crossfade
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        # Resamplers on CUDA
        self.resampler = tat.Resample(
            orig_freq=self.samplerate, new_freq=16000, dtype=torch.float32
        ).to(self.device)
        self.resampler2 = None
        if self.rvc.tgt_sr != self.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.samplerate,
                dtype=torch.float32,
            ).to(self.device)

        # Inter-thread communication:
        # - _input_queue: mic blocks from input callback → worker thread
        # - _output_queue: processed blocks from worker thread → output callback
        self._input_queue = collections.deque(maxlen=8)
        self._output_queue = collections.deque(maxlen=8)
        self._input_event = threading.Event()

        # Open audio streams — separate in/out to avoid PortAudio
        # "Illegal combination of I/O devices" when devices are on
        # different host APIs (common with PipeWire virtual devices)
        self._running = True
        self._input_stream = sd.InputStream(
            callback=self._input_callback,
            blocksize=self.block_frame,
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            device=input_device_idx,
        )
        self._output_stream = sd.OutputStream(
            callback=self._output_callback,
            blocksize=self.block_frame,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            device=output_device_idx,
        )

        # Start worker thread for inference (keeps audio callbacks fast)
        self._worker = threading.Thread(target=self._infer_loop, daemon=True)
        self._worker.start()

        self._input_stream.start()
        self._output_stream.start()
        return "Running (sr=%d, block=%dms)" % (
            self.samplerate,
            int(block_time * 1000),
        )

    def stop(self):
        self._running = False
        # Wake worker so it can exit
        if hasattr(self, "_input_event"):
            self._input_event.set()
        if hasattr(self, "_worker") and self._worker is not None:
            self._worker.join(timeout=2)
            self._worker = None
        for s in ("_input_stream", "_output_stream"):
            stream = getattr(self, s, None)
            if stream is not None:
                stream.abort()
                stream.close()
                setattr(self, s, None)
        return "Stopped"

    def is_running(self):
        return self._running

    def update_pitch(self, new_pitch):
        if self.rvc is not None:
            self.rvc.change_key(new_pitch)

    def update_index_rate(self, new_rate):
        if self.rvc is not None:
            self.rvc.change_index_rate(new_rate)

    def get_status(self):
        if not self._running:
            return "Stopped"
        return "Running (latency: %dms)" % int(self._last_infer_ms)

    def _input_callback(self, indata, frames, times, status):
        """Capture mic block and hand off to worker thread. Must be fast."""
        if not self._running:
            return
        # Copy the mono audio data (cheap CPU op)
        self._input_queue.append(indata[:, 0].copy())
        self._input_event.set()

    def _output_callback(self, outdata, frames, times, status):
        """Write the next processed block to the output device, or silence."""
        if not self._running or not self._output_queue:
            outdata[:] = 0
            return
        try:
            outdata[:] = self._output_queue.popleft()
        except IndexError:
            outdata[:] = 0

    def _infer_loop(self):
        """Worker thread: consumes mic blocks, runs RVC inference, produces output."""
        while self._running:
            # Wait for input data
            self._input_event.wait(timeout=0.5)
            self._input_event.clear()

            if not self._running:
                break

            # Drain all pending input blocks into the ring buffer
            while self._input_queue:
                try:
                    mono = self._input_queue.popleft()
                except IndexError:
                    break
                self.input_wav[: -self.block_frame] = self.input_wav[
                    self.block_frame :
                ].clone()
                self.input_wav[-mono.shape[0] :] = torch.from_numpy(mono).to(
                    self.device
                )
                self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                    self.block_frame_16k :
                ].clone()
                self.input_wav_res[-160 * (mono.shape[0] // self.zc + 1) :] = (
                    self.resampler(
                        self.input_wav[-mono.shape[0] - 2 * self.zc :]
                    )[160:]
                )

            # Run inference on the latest accumulated input
            start_time = time.perf_counter()
            try:
                infer_wav = self.rvc.infer(
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.return_length,
                    self.f0method,
                )

                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)

                # SOLA crossfade for smooth transitions (CUDA)
                conv_input = infer_wav[
                    None, None, : self.sola_buffer_frame + self.sola_search_frame
                ]
                cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
                cor_den = torch.sqrt(
                    F.conv1d(
                        conv_input**2,
                        torch.ones(
                            1, 1, self.sola_buffer_frame, device=self.device
                        ),
                    )
                    + 1e-8
                )
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
                infer_wav = infer_wav[sola_offset:]
                infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
                infer_wav[: self.sola_buffer_frame] += (
                    self.sola_buffer * self.fade_out_window
                )
                self.sola_buffer[:] = infer_wav[
                    self.block_frame : self.block_frame + self.sola_buffer_frame
                ]

                # Convert to numpy and enqueue for output
                result = (
                    infer_wav[: self.block_frame]
                    .repeat(self.channels, 1)
                    .t()
                    .cpu()
                    .numpy()
                )
                self._output_queue.append(result)
            except Exception:
                import traceback
                traceback.print_exc()

            self._last_infer_ms = (time.perf_counter() - start_time) * 1000

    def _get_device_channels(self, device_idx):
        info = sd.query_devices(device_idx)
        return min(int(info["max_output_channels"]), 2)

    def _cleanup(self):
        if self._running:
            self.stop()
