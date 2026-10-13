"""Host tests of signal math and camera-loop contracts, not board validation."""

import contextlib
import csv
import io
import math
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import k210_rppg_basic as basic

try:
    import numpy as np
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None


class SignalTests(unittest.TestCase):
    def test_known_pulses_with_trend_noise_and_irregular_timestamps(self):
        for bpm in (60, 72, 93, 120, 150, 180):
            with self.subTest(bpm=bpm):
                rng = random.Random(210)
                window = basic.GreenWindow()
                t = 0
                while t < 12000:
                    green = 25 + 0.0002 * t + 0.4 * math.sin(2 * math.pi * bpm * t / 60000)
                    window.push(t, green + rng.uniform(-0.05, 0.05))
                    t += rng.randint(60, 85)
                actual, state = window.estimate()
                self.assertEqual(state, "ESTIMATE")
                self.assertLessEqual(abs(actual - bpm), 3.0)

    def test_dark_weak_signal_has_no_brightness_or_amplitude_gate(self):
        window = basic.GreenWindow()
        for t in range(0, basic.WINDOW_MS + 2001, 100):
            window.push(t, 8 + 0.00001 * math.sin(2 * math.pi * 1.2 * t / 1000))
        self.assertEqual(window.estimate(), (72.0, "ESTIMATE"))

    def test_noise_still_produces_candidate_without_a_quality_gate(self):
        rng = random.Random(42)
        values = [20 + rng.uniform(-1, 1) for _ in range(101)]
        bpm, status = basic.spectral_bpm(values, 20, 42, 240)
        self.assertEqual(status, "ESTIMATE")
        self.assertTrue(42 <= bpm <= 240)

    def test_warmup_constant_and_linear_signals_do_not_invent_bpm(self):
        window = basic.GreenWindow()
        window.push(0, 10)
        self.assertEqual(window.estimate(), (None, "WARMUP"))
        for values in ([10.0] * 101, [10 + i * 0.1 for i in range(101)]):
            self.assertEqual(basic.spectral_bpm(values, 20, 42, 240), (None, "NO SIGNAL"))

    def test_color_step_and_missed_frames_preserve_real_history(self):
        window = basic.GreenWindow(window_ms=5000)
        for t in range(0, 4001, 100):
            window.push(t, 10 + math.sin(t / 100.0))
        original = list(window.points)
        window.push(4100, None)
        self.assertEqual(window.points, original)
        window.push(4200, 80)
        self.assertEqual(window.points[:-1], original)
        for t in range(4300, 6001, 100):
            window.push(t, 80 + math.sin(t / 100.0))
        before = list(window.points)
        self.assertEqual(window.estimate()[1], "ESTIMATE")
        self.assertEqual(window.points, before)  # Processing never changes raw G.

    def test_short_gap_recovery_and_natural_expiry_after_long_absence(self):
        window = basic.GreenWindow(window_ms=5000)
        for t in range(0, 8001, 100):
            green = None if 4500 <= t <= 4700 else 20 + math.sin(2 * math.pi * 1.2 * t / 1000)
            window.push(t, green)
        self.assertEqual(window.estimate(), (72.0, "ESTIMATE"))
        window.push(14000, None)
        self.assertEqual(window.points, [])
        window.push(14100, 20)
        self.assertEqual(window.estimate(), (None, "WARMUP"))
        # Even if no missing-frame calls occur, old history expires on return.
        window.push(22000, 22)
        self.assertEqual(window.points, [(22000, 22)])

    def test_interpolation_and_memory_capacity(self):
        self.assertEqual(basic.resample([(0, 1), (200, 5), (300, 7)], 50, 250, 3), [2, 4, 6])
        window = basic.GreenWindow(capacity=20)
        for t in range(0, 10000, 10):
            window.push(t, 20)
        self.assertEqual(len(window.points), 20)
        self.assertEqual(window.estimate(), (None, "BUFFER LIMIT"))

    def test_search_respects_nyquist(self):
        values = [math.sin(i) for i in range(20)]
        bpm, _ = basic.spectral_bpm(values, 3, 42, 240)
        self.assertLessEqual(bpm, 90)
        self.assertEqual(basic.spectral_bpm(values, 1, 42, 240), (None, "LOW RATE"))

    @unittest.skipIf(scipy_signal is None, "SciPy is optional and only used on the host")
    def test_filter_matches_independent_scipy_reference(self):
        rng = random.Random(11)
        for fs in (3.0, 6.0, 8.0, 10.0, 20.0):
            with self.subTest(fs=fs):
                values = [20 + math.sin(i * 0.23) + rng.uniform(-1, 1) for i in range(121)]
                original = list(values)
                sections = [scipy_signal.butter(2, 0.7, "highpass", fs=fs, output="sos")]
                if fs > 8:
                    sections.append(scipy_signal.butter(2, 4, "lowpass", fs=fs, output="sos"))
                expected = scipy_signal.sosfiltfilt(np.vstack(sections), values,
                    padtype="odd", padlen=min(len(values) - 1, int(2 * fs)))
                actual = basic.bandpass_filter(values, fs, 0.7, 4)
                self.assertLess(max(abs(a - e) for a, e in zip(actual, expected)), 1e-10)
                self.assertEqual(values, original)

    def test_filter_suppresses_outside_band_and_retains_pulse(self):
        fs = 20.0
        amplitudes = []
        for hz in (0.2, 1.2, 6.0):
            values = [math.sin(2 * math.pi * hz * i / fs) for i in range(801)]
            filtered = basic.bandpass_filter(values, fs, 0.7, 4.0)
            interior = range(200, 601)  # Ignore finite-window edge effects.
            cosine = sum(filtered[i] * math.cos(2 * math.pi * hz * i / fs) for i in interior)
            sine = sum(filtered[i] * math.sin(2 * math.pi * hz * i / fs) for i in interior)
            amplitudes.append(2 * math.sqrt(cosine * cosine + sine * sine) / len(interior))
        self.assertLess(amplitudes[0], 0.02)
        self.assertGreater(amplitudes[1], 0.8)
        self.assertLess(amplitudes[2], 0.1)

    def test_filter_comparison_switch_and_higher_pulse_rates(self):
        fs = 20.0
        for bpm in (48, 72, 120, 180, 210, 234):
            values = [25 + 0.01 * i + math.sin(2 * math.pi * bpm * i / (60 * fs))
                      for i in range(201)]
            for enabled in (False, True):
                actual, status = basic.spectral_bpm(values, fs, 42, 240, enabled)
                self.assertEqual(status, "ESTIMATE")
                self.assertLessEqual(abs(actual - bpm), 3)


class PosTests(unittest.TestCase):
    @staticmethod
    def color(t, bpm=72.0, interference=True):
        pulse = math.sin(2 * math.pi * bpm * t / 60000)
        light = 1 + (0.08 * math.sin(2 * math.pi * 102 * t / 60000) if interference else 0)
        return tuple(base * light * (1 + amplitude * pulse)
                     for base, amplitude in zip((50, 40, 30), (0.0033, 0.0077, 0.0053)))

    @unittest.skipIf(scipy_signal is None, "NumPy reference is optional on the host")
    def test_pos_matches_matrix_reference_including_last_subwindow(self):
        # Independently vectorized Algorithm 1, not the board implementation.
        for fs in (8.0, 12.7, 20.0):
            colors = np.array([self.color(i * 1000 / fs) for i in range(128)]).T
            original = colors.copy()
            length = math.ceil(1.6 * fs)
            expected = np.zeros(128)
            projection = np.array([[0, 1, -1], [-2, 1, 1]])
            for end in range(length, 129):
                c = colors[:, end-length:end]
                s = projection @ (c / c.mean(axis=1, keepdims=True))
                h = s[0] + np.std(s[0]) / np.std(s[1]) * s[1]
                expected[end-length:end] += h - np.mean(h)
            actual, state = basic.pos_signal(*colors.tolist(), fs)
            self.assertEqual(state, "READY")
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
            np.testing.assert_array_equal(colors, original)

    def test_common_brightness_interference_same_window_comparison(self):
        # Synthetic 72 BPM pulse + stronger 102 BPM multiplicative lighting.
        # This test makes no claim about real camera/skin performance.
        for step in (50, 100, 125):
            window = basic.RGBWindow()
            rng = random.Random(21)
            for t in range(0, 12001, step):
                rgb = tuple(c + rng.uniform(-0.005, 0.005) for c in self.color(t))
                window.push(t, rgb)
            original = list(window.points)
            self.assertEqual(window.estimate(), (72.0, "ESTIMATE"))
            self.assertEqual(window.green_bpm, 102.0)
            self.assertEqual(window.compare_t_ms, original[-1][0])
            self.assertEqual(window.points, original)

    def test_pulse_rates_with_irregular_times_and_missed_detections(self):
        for bpm in (48, 72, 120, 180, 234):
            window = basic.RGBWindow()
            rng, t = random.Random(210), 0
            while t < 12500:
                window.push(t, None if 8500 < t < 8650 else self.color(t, bpm, False))
                t += rng.randint(60, 85)
            actual, state = window.estimate()
            self.assertEqual(state, "ESTIMATE")
            self.assertLessEqual(abs(actual - bpm), 3.0)

    def test_constant_or_common_mode_only_does_not_invent_pulse(self):
        for varying in (False, True):
            colors = [(50 * factor, 40 * factor, 30 * factor)
                      for factor in (1 + (0.08 * math.sin(i) if varying else 0)
                                     for i in range(101))]
            signal, state = basic.pos_signal(*zip(*colors), 10.0)
            self.assertEqual(state, "READY")
            self.assertEqual(basic.spectral_bpm(signal, 10, 42, 240), (None, "NO SIGNAL"))

    def test_degenerate_projection_zero_channel_and_short_input(self):
        self.assertEqual(basic.pos_signal([1]*2, [1]*2, [1]*2, 10), (None, "WARMUP"))
        self.assertEqual(basic.pos_signal([0]*20, [2]*20, [1]*20, 10), (None, "NO RGB"))
        perturbation = [0.01, -0.01] * 8
        self.assertEqual(basic.pos_signal([50]*16, [40*(1+s) for s in perturbation],
                         [30*(1-s) for s in perturbation], 10), (None, "POS DEGENERATE"))
        with self.assertRaises(ValueError):
            basic.pos_signal([1], [1]*10, [1], 10)

    def test_switch_disables_unused_method_and_preserves_green_baseline(self):
        baseline = basic.GreenWindow()
        window = basic.RGBWindow(method="GREEN", compare=False)
        for t in range(0, 12001, 100):
            rgb = self.color(t)
            baseline.push(t, rgb[1])
            window.push(t, rgb)
        with patch.object(basic, "pos_signal", side_effect=AssertionError("POS should be disabled")):
            self.assertEqual(window.estimate(), baseline.estimate())
        self.assertIsNone(window.pos_bpm)
        self.assertEqual(window.pos_state, "DISABLED")
        window.method = "POS"
        self.assertEqual(window.estimate(), (72.0, "ESTIMATE"))
        self.assertIsNone(window.green_bpm)
        self.assertEqual(window.green_state, "DISABLED")

    def test_invalid_pos_does_not_silently_fall_back_or_clear_history(self):
        window = basic.RGBWindow()
        for t in range(0, 12001, 100):
            window.push(t, (0, 20 + math.sin(2 * math.pi * 1.2 * t / 1000), 10))
        original = list(window.points)
        self.assertEqual(window.estimate(), (None, "NO RGB"))
        self.assertEqual(window.green_bpm, 72.0)
        self.assertEqual(window.points, original)
        window.clear_results("NO FACE")
        self.assertEqual(window.compare_t_ms, -1)
        self.assertIsNone(window.green_bpm)
        self.assertEqual(window.points, original)
        window.push(12500, None)
        self.assertTrue(window.points)
        window.push(25000, None)
        self.assertEqual(window.points, [])


class Face:
    def __init__(self, box):
        self.box = box

    def rect(self):
        return self.box

    def w(self):
        return self.box[2]

    def h(self):
        return self.box[3]


class BoardLoopTests(unittest.TestCase):
    def test_largest_face_and_forehead_geometry(self):
        small, large = Face((10, 20, 20, 30)), Face((50, 40, 100, 120))
        self.assertIs(basic.choose_face([small, large]), large)
        self.assertIsNone(basic.choose_face(None))
        self.assertEqual(basic.forehead_rect(large.rect()), (80, 52, 40, 18))
        self.assertEqual(basic.forehead_rect((280, 220, 100, 120)), (310, 232, 10, 8))
        self.assertIsNone(basic.forehead_rect((400, 300, 50, 60)))

    def run_board(self, fail_frame=None):
        clock = {"frame": -1, "ms": 0, "deinit": False, "display": 0}
        tester = self

        class Image:
            def __init__(self):
                self.overlaid = False

            def get_pixel(self, x, y):
                tester.assertFalse(self.overlaid, "RGB read after drawing")
                return (30.0, 20 + math.sin(2 * math.pi * 1.2 * clock["ms"] / 1000), 10.0)

            def draw_rectangle(self, *args, **kwargs):
                self.overlaid = True

            def draw_string(self, *args, **kwargs):
                self.overlaid = True

            def draw_line(self, *args, **kwargs):
                self.overlaid = True

            def width(self):
                return 320

            def height(self):
                return 240

        def snapshot():
            clock["frame"] += 1
            clock["ms"] = clock["frame"] * 100
            return Image()

        def detect(task, img):
            if clock["frame"] == fail_frame:
                raise RuntimeError("detector failed")
            if clock["frame"] in (12, 13, 105):
                return None
            return [Face((50 + clock["frame"] % 4, 40, 100, 120))]

        def ignore(*args, **kwargs):
            pass

        def deinit(task):
            clock["deinit"] = True

        def display(img):
            clock["display"] += 1

        modules = {
            "sensor": types.SimpleNamespace(RGB565=1, QVGA=2, reset=ignore,
                set_pixformat=ignore, set_framesize=ignore, set_vflip=ignore,
                set_hmirror=ignore, skip_frames=ignore, snapshot=snapshot),
            "lcd": types.SimpleNamespace(init=ignore, display=display),
            "KPU": types.SimpleNamespace(load=lambda addr: object(), init_yolo2=ignore,
                run_yolo2=detect, deinit=deinit),
            "time": types.SimpleNamespace(ticks_ms=lambda: clock["ms"] % 2048,
                ticks_diff=lambda a, b: (a - b + 1024) % 2048 - 1024),
        }
        output = io.StringIO()
        with patch.dict(sys.modules, modules), patch.object(basic, "RUN_SECONDS", 12), contextlib.redirect_stdout(output):
            if fail_frame is None:
                basic.main()
            else:
                with self.assertRaisesRegex(RuntimeError, "detector failed"):
                    basic.main()
        self.assertTrue(clock["deinit"])
        return output.getvalue()

    def test_full_loop_dynamic_roi_missing_face_timestamps_and_cleanup(self):
        output = self.run_board()
        rows = list(csv.DictReader(line for line in output.splitlines() if not line.startswith("#")))
        self.assertEqual(len(rows), 120)
        self.assertEqual(rows[-1]["t_ms"], "11900")  # Multiple mock clock wraps.
        self.assertNotEqual(rows[0]["roi_x"], rows[1]["roi_x"])
        self.assertEqual(rows[12]["g"], "-1.000")
        self.assertEqual(rows[12]["samples"], rows[11]["samples"])
        self.assertEqual(rows[90]["bpm"], "-1.0")
        self.assertEqual(rows[90]["status"], "WARMUP")
        self.assertEqual(rows[100]["bpm"], "72.0")
        self.assertEqual(rows[105]["bpm"], "-1.0")
        self.assertEqual(rows[105]["status"], "NO FACE")
        self.assertEqual(rows[110]["bpm"], "72.0")
        self.assertEqual(rows[110]["bpm_t_ms"], "11000")
        self.assertEqual(rows[0]["fps"], "-1.0")
        self.assertEqual(rows[110]["fps"], "10.0")
        summaries = [line for line in output.splitlines() if line.startswith("# HR_RESULT;")]
        last = dict(part.strip().split("=", 1) for part in summaries[-1].split(";")[1:])
        self.assertEqual(last["fps"], rows[110]["fps"])
        self.assertEqual(last["bpm"], rows[110]["bpm"])
        self.assertEqual(last["bpm_t_ms"], rows[110]["bpm_t_ms"])
        self.assertEqual(len(rows[0]), 23)
        for key in ("method", "pos_bpm", "green_bpm", "pos_state", "green_state", "compare_t_ms"):
            self.assertEqual(last[key], rows[110][key])
        self.assertEqual(last["method"], "POS")
        self.assertEqual(last["pos_bpm"], "72.0")
        self.assertEqual(last["green_bpm"], "72.0")
        self.assertEqual(last["compare_t_ms"], "11000")
        for row in rows[105:110]:
            self.assertEqual(row["pos_bpm"], "-1.0")
            self.assertEqual(row["green_bpm"], "-1.0")
            self.assertEqual(row["compare_t_ms"], "-1")
        self.assertEqual(rows[106]["pos_state"], "WARMUP")
        self.assertEqual(output.count("# CAPABILITIES;"), 1)
        self.assertIn("# capture_stop; reason=duration", output)

    def test_runtime_failure_is_visible_and_releases_kpu(self):
        output = self.run_board(fail_frame=2)
        self.assertIn("# error=detector failed", output)
        self.assertIn("# capture_stop; reason=error", output)

    def test_legacy_ide_transport_guard(self):
        source = Path(basic.__file__).read_bytes()
        self.assertTrue(source.endswith(b"\n# IDE_EOF_GUARD"))
        self.assertFalse(any(byte < 32 and byte not in (9, 10, 13) for byte in source))


if __name__ == "__main__":
    unittest.main()
