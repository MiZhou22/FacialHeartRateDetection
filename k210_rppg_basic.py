"""Single-file MaixPy v1 demo: YOLOv2 -> forehead mean RGB -> POS/GREEN.

Run the whole file in MaixPy IDE. No extra module or new firmware is needed.
Assume one visible face. Every frame uses its CURRENT detection, without KLT.
Positions are QVGA pixels; RGB565 channels are expanded values in 0..255.
Times are snapshot-return times in ms, not exposure timestamps. EXP BPM is an
unvalidated frequency candidate. Missing faces skip sampling, never reuse RGB.
"""

import math

VERSION = "1.2.0-basic-rppg"
MODEL_ADDR = 0x300000
CONFIDENCE = 0.30
NMS = 0.30
ANCHORS = (1.889, 2.5245, 2.9465, 3.94056, 3.99987,
           5.3658, 5.155437, 6.92275, 6.718375, 9.01025)
FOREHEAD = (0.30, 0.10, 0.40, 0.15)  # x/y offset and w/h, relative to face.
SAMPLE_STEP = 1                       # Average every forehead pixel; >1 subsamples.
WINDOW_MS = 10000                     # Longer observation for frequency estimation.
PLOT_WINDOW_MS = 5000                 # Keep the small live trace readable.
PLOT_RECALCULATE = True               # False freezes each new point's height.
PLOT_Y_LIMIT = 1.0                    # Fixed +/- baseline-removed G units, 0..255.
PLOT_BASELINE_MS = 1000               # Fixed-point mode baseline time constant, ms.
UPDATE_MS = 1000                      # Compute/display a new BPM once a second.
HR_METHOD = "POS"                    # "GREEN" restores the single-channel method.
COMPARE_METHODS = True               # Same-window comparison; False saves HR work.
POS_WINDOW_MS = 1600                  # POS normalization/overlap subwindow, ms.
BANDPASS_ENABLED = True               # False bypasses filtering for manual A/B runs.
MIN_BPM = 42.0
MAX_BPM = 240.0
BPM_STEP = 3.0                        # Search spacing, NOT measurement accuracy.
MAX_SAMPLES = 256                     # Bounded signal history, no image history.
LOG_SAMPLES = True                    # Stream real RGB to serial; no capture buffer.
RUN_SECONDS = 0                       # 0: run until stopped; >0: timed experiment.


def choose_face(detections):
    """Select the largest box; this basic version assumes a single person."""
    best, area = None, 0
    for face in detections or ():
        size = face.w() * face.h()
        if face.w() > 0 and face.h() > 0 and size > area:
            best, area = face, size
    return best


def forehead_rect(box, width=320, height=240):
    """Return a clipped (x,y,w,h) ROI, or None if it has no image pixels."""
    x, y, w, h = box
    x0 = int(x + w * FOREHEAD[0])
    y0 = int(y + h * FOREHEAD[1])
    x1 = int(x + w * (FOREHEAD[0] + FOREHEAD[2]))
    y1 = int(y + h * (FOREHEAD[1] + FOREHEAD[3]))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def rgb_mean(img, roi):
    """Read current pixels BEFORE drawing; return spatial means (R,G,B)."""
    x, y, w, h = roi
    red = green = blue = count = 0
    get_pixel = img.get_pixel
    for py in range(y, y + h, SAMPLE_STEP):
        for px in range(x, x + w, SAMPLE_STEP):
            r, g, b = get_pixel(px, py)
            red += r
            green += g
            blue += b
            count += 1
    return red / count, green / count, blue / count


def resample(points, start_ms, end_ms, count, channel=1):
    """Interpolate a tuple channel to count equally spaced samples.

    Index 0 is time_ms; index 1 is G (RGBWindow additionally uses R=2, B=3).
    Both endpoints must be covered by real samples. Missing intervals inside
    the window are bridged linearly, without a gap-quality rejection rule.
    This approximation can distort a signal with many missed detections.
    """
    values, index = [], 0
    for i in range(count):
        target = start_ms + (end_ms - start_ms) * i / (count - 1)
        while index + 1 < len(points) - 1 and points[index + 1][0] < target:
            index += 1
        left, right = points[index], points[index + 1]
        weight = (target - left[0]) / (right[0] - left[0])
        values.append(left[channel] + weight * (right[channel] - left[channel]))
    return values


def pos_signal(red, green, blue, sample_hz):
    """Three uniform N-vectors (RGB means) -> dimensionless POS N-vector.

    Independent implementation of Wang et al., Algorithm 1:
    https://doi.org/10.1109/TBME.2016.2609282
    Normalize each channel by its local mean; project G-B and -2R+G+B;
    combine using the standard-deviation ratio, center, then overlap-add.
    No NumPy, matrix inverse, quality gate or automatic GREEN fallback.
    Downstream filtering is our existing pipeline, separate from POS core.
    """
    n = len(green)
    if len(red) != n or len(blue) != n or sample_hz <= 0 or POS_WINDOW_MS <= 0:
        raise ValueError("invalid POS input shape/rate")
    length = max(3, int(math.ceil(POS_WINDOW_MS * sample_hz / 1000.0)))
    if n < length:
        return None, "WARMUP"
    pulse = [0.0] * n
    for start in range(n - length + 1):
        end = start + length
        mr = sum(red[start:end]) / length
        mg = sum(green[start:end]) / length
        mb = sum(blue[start:end]) / length
        if mr <= 0 or mg <= 0 or mb <= 0:
            return None, "NO RGB"  # Cannot normalize a zero/absent channel.
        s1, s2 = [], []
        for i in range(start, end):
            r, g, b = red[i] / mr, green[i] / mg, blue[i] / mb
            s1.append(g - b)
            s2.append(-2.0 * r + g + b)
        mean1, mean2 = sum(s1) / length, sum(s2) / length
        energy1 = sum((value - mean1) ** 2 for value in s1)
        energy2 = sum((value - mean2) ** 2 for value in s2)
        # Numerical zero guard for the ratio, not a signal-quality threshold.
        if energy2 <= 1e-24:
            if energy1 <= 1e-24:
                continue
            return None, "POS DEGENERATE"
        alpha = math.sqrt(energy1 / energy2)
        for i in range(length):
            pulse[start + i] += s1[i] - mean1 + alpha * (s2[i] - mean2)
    return pulse, "READY"


def butterworth_section(values, sample_hz, cutoff_hz, highpass):
    """Second-order Butterworth section, steady initial state, O(N) work.

    Bilinear-transform coefficients; frequencies are Hz, Q=1/sqrt(2).
    Direct-form II transposed state is local to this temporary signal.
    """
    angle = 2.0 * math.pi * cutoff_hz / sample_hz
    cosine = math.cos(angle)
    alpha = math.sin(angle) / math.sqrt(2.0)
    norm = 1.0 + alpha
    b0 = (1.0 + cosine if highpass else 1.0 - cosine) / (2.0 * norm)
    b1, b2 = (-2.0 if highpass else 2.0) * b0, b0
    a1, a2 = -2.0 * cosine / norm, (1.0 - alpha) / norm
    steady = 0.0 if highpass else values[0]
    state1 = steady - b0 * values[0]
    state2 = b2 * values[0] - a2 * steady
    result = []
    for value in values:
        output = b0 * value + state1
        state1 = b1 * value - a1 * output + state2
        state2 = b2 * value - a2 * output
        result.append(output)
    return result


def bandpass_filter(values, sample_hz, low_hz, high_hz):
    """Forward/backward high-pass + low-pass, with 2 s odd reflection.

    Padding reduces edge transients; it never adds observations to history.
    If the high corner is at/above Nyquist, omit the low-pass section; there
    is no representable band above Nyquist. This cannot undo sampling aliasing.
    """
    if not 0 < low_hz < sample_hz * 0.5 or high_hz <= low_hz:
        raise ValueError("invalid filter frequencies")
    pad = min(len(values) - 1, int(2.0 * sample_hz))
    signal = ([2.0 * values[0] - values[i] for i in range(pad, 0, -1)] +
              list(values) +
              [2.0 * values[-1] - values[-i - 1] for i in range(1, pad + 1)])
    for direction in range(2):
        signal = butterworth_section(signal, sample_hz, low_hz, True)
        if high_hz < sample_hz * 0.5:
            signal = butterworth_section(signal, sample_hz, high_hz, False)
        signal.reverse()
    return signal[pad:pad + len(values)]


def spectral_bpm(values, sample_hz, low_bpm, high_bpm, filter_enabled=None):
    """N-vector signal -> detrend -> bandpass -> Hann -> strongest band peak.

    Goertzel evaluates the spectrum without NumPy/FFT firmware dependencies.
    No peak-ratio, amplitude-quality, motion or repeated-confirmation gate.
    """
    n = len(values)
    if n < 3:
        return None, "WARMUP"
    mean = sum(values) / n
    center = (n - 1) * 0.5
    denominator = sum((i - center) ** 2 for i in range(n))
    slope = sum((i - center) * (values[i] - mean) for i in range(n)) / denominator
    signal = [values[i] - mean - slope * (i - center) for i in range(n)]
    # Numerical zero only: constant/linear input has no periodic information.
    if sum(value * value for value in signal) <= 1e-18:
        return None, "NO SIGNAL"
    upper = min(high_bpm, sample_hz * 30.0)  # Nyquist, in BPM.
    if upper <= low_bpm:
        return None, "LOW RATE"
    if filter_enabled is None:
        filter_enabled = BANDPASS_ENABLED
    if filter_enabled:
        signal = bandpass_filter(signal, sample_hz, low_bpm / 60.0, high_bpm / 60.0)
    signal = [signal[i] * (0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)))
              for i in range(n)]
    best_bpm, best_power = None, -1.0
    bins = int((upper - low_bpm) / BPM_STEP) + 1
    for index in range(bins):
        bpm = low_bpm + index * BPM_STEP
        coefficient = 2.0 * math.cos(2.0 * math.pi * (bpm / 60.0) / sample_hz)
        previous = previous2 = 0.0
        for value in signal:
            current = value + coefficient * previous - previous2
            previous2, previous = previous, current
        power = previous * previous + previous2 * previous2 - coefficient * previous * previous2
        if power > best_power:
            best_bpm, best_power = bpm, power
    return best_bpm, "ESTIMATE"


class GreenWindow:
    """Bounded rolling real samples. No motion/ROI/color-triggered resets."""
    def __init__(self, window_ms=WINDOW_MS, capacity=MAX_SAMPLES):
        self.window_ms, self.capacity = window_ms, capacity
        self.points = []

    def push(self, t_ms, green):
        cutoff = t_ms - self.window_ms
        # Retain one boundary sample for interpolation, but expire ALL history
        # when its newest sample has left the window. Do this before appending.
        while len(self.points) > 1 and self.points[1][0] <= cutoff:
            del self.points[0]
        if self.points and self.points[-1][0] < cutoff:
            del self.points[:]
        if green is not None:
            if self.points and t_ms <= self.points[-1][0]:
                raise ValueError("sample timestamps must increase")
            self.points.append((t_ms, green))
            if len(self.points) > self.capacity:
                del self.points[0]

    def uniform(self, channel=1):
        """Return (uniform values, sample_hz, state), without changing history."""
        points = self.points
        if len(points) < 3 or points[-1][0] - points[0][0] < self.window_ms:
            return None, 0.0, "BUFFER LIMIT" if len(points) >= self.capacity else "WARMUP"
        end = points[-1][0]
        count = min(128, len(points))
        values = resample(points, end - self.window_ms, end, count, channel)
        sample_hz = (count - 1) * 1000.0 / self.window_ms
        return values, sample_hz, "READY"

    def estimate(self):
        values, sample_hz, state = self.uniform()
        if values is None:
            return None, state
        return spectral_bpm(values, sample_hz, MIN_BPM, MAX_BPM)


class RGBWindow(GreenWindow):
    """One shared raw history: (t_ms,G,R,B), preserving baseline G at index 1.

    POS/GREEN results use identical timestamps, resampling and spectral math.
    Results persist between estimates; missing faces clear results, not samples.
    compare_t_ms is the last evaluated window end, even if no peak was found.
    """
    def __init__(self, method=HR_METHOD, compare=COMPARE_METHODS,
                 window_ms=WINDOW_MS, capacity=MAX_SAMPLES):
        GreenWindow.__init__(self, window_ms, capacity)
        if method not in ("POS", "GREEN"):
            raise ValueError("HR_METHOD must be POS or GREEN")
        self.method, self.compare = method, compare
        self.clear_results("WARMUP")

    def clear_results(self, state):
        self.pos_bpm, self.green_bpm = None, None
        self.pos_state = state if self.compare or self.method == "POS" else "DISABLED"
        self.green_state = state if self.compare or self.method == "GREEN" else "DISABLED"
        self.compare_t_ms = -1

    def push(self, t_ms, rgb):
        GreenWindow.push(self, t_ms, None if rgb is None else rgb[1])
        if rgb is not None:
            self.points[-1] = (t_ms, rgb[1], rgb[0], rgb[2])

    def estimate(self):
        green, sample_hz, state = self.uniform()
        self.clear_results(state)
        if green is None:
            return None, state
        self.compare_t_ms = self.points[-1][0]
        if self.compare or self.method == "GREEN":
            self.green_bpm, self.green_state = spectral_bpm(green, sample_hz, MIN_BPM, MAX_BPM)
        if self.compare or self.method == "POS":
            red, _, _ = self.uniform(2)
            blue, _, _ = self.uniform(3)
            pulse, self.pos_state = pos_signal(red, green, blue, sample_hz)
            if pulse is not None:
                self.pos_bpm, self.pos_state = spectral_bpm(pulse, sample_hz, MIN_BPM, MAX_BPM)
        return ((self.pos_bpm, self.pos_state) if self.method == "POS" else
                (self.green_bpm, self.green_state))


class GreenTrace(GreenWindow):
    """Display-only causal baseline removal; stored point heights never change.

    Update baseline on every real sample: b += (1-exp(-dt/tau))*(G-b).
    Store G-b at most once per 1/64 plot span, giving at most 65 visible points.
    Decimation happens on arrival, so redrawing never reselects old vertices.
    This separate small buffer is never used to estimate heart rate.
    """
    def __init__(self):
        if PLOT_BASELINE_MS <= 0 or PLOT_WINDOW_MS <= 0:
            raise ValueError("plot times must be positive")
        GreenWindow.__init__(self, PLOT_WINDOW_MS, 66)
        self.baseline, self.last_ms = None, None

    def push(self, t_ms, green):
        value = None
        if green is not None:
            if self.last_ms is not None and t_ms <= self.last_ms:
                raise ValueError("trace timestamps must increase")
            if self.baseline is None:
                self.baseline = green
            else:
                alpha = 1.0 - math.exp(-(t_ms - self.last_ms) / PLOT_BASELINE_MS)
                self.baseline += alpha * (green - self.baseline)
            self.last_ms = t_ms
            if not self.points or t_ms - self.points[-1][0] >= PLOT_WINDOW_MS / 64.0:
                value = green - self.baseline
        GreenWindow.push(self, t_ms, value)


def draw_green_trace(img, points, now_ms, fixed_points):
    """Scroll G with fixed time/amplitude scales, in either display mode.

    Default: refit the recent raw G window's linear trend on every frame.
    Optional: immutable causal GreenTrace heights, without retroactive edits.
    New points enter on the right and leave on the left; never sweep/reset.
    This is a camera color waveform, not a validated heartbeat or ECG trace.
    """
    if PLOT_Y_LIMIT <= 0:
        raise ValueError("PLOT_Y_LIMIT must be positive")
    plot_width = img.width() // 4
    plot_top = img.height() - 48
    mid_y = plot_top + 24
    # Draw directly onto each fresh camera frame, without a filled background.
    img.draw_line((4, mid_y, plot_width - 5, mid_y), color=(70, 70, 70))
    start_ms = now_ms - PLOT_WINDOW_MS
    if PLOT_RECALCULATE:
        # RGBWindow stores (t_ms,G,R,B); only the first two fields are drawn.
        recent = [(p[0] - start_ms, p[1]) for p in points if start_ms <= p[0] <= now_ms]
        n = len(recent)
        if n < 3:
            return
        mean_t = sum(t for t, g in recent) / n
        mean_g = sum(g for t, g in recent) / n
        variance_t = sum((t - mean_t) ** 2 for t, g in recent)
        slope = sum((t - mean_t) * (g - mean_g) for t, g in recent) / variance_t
        trace = [(t, g - mean_g - slope * (t - mean_t)) for t, g in recent]
        stride = max(1, (n + 63) // 64)
        selected = trace[::stride]
        if selected[-1] != trace[-1]:
            selected.append(trace[-1])
    else:
        selected = [(t - start_ms, g) for t, g in fixed_points if start_ms <= t <= now_ms]
    previous = None
    for t, value in selected:
        # Clip display excursions only; do not rescale the window or raw data.
        shown = max(-PLOT_Y_LIMIT, min(PLOT_Y_LIMIT, value))
        point = (4 + int((plot_width - 9) * t / PLOT_WINDOW_MS),
                 mid_y - int(18 * shown / PLOT_Y_LIMIT))
        if previous is not None:
            img.draw_line(previous + point, color=(0, 255, 255))
        previous = point


def report_capabilities(img):
    """Report available APIs once; no tracking, inverse or firmware change."""
    try:
        from ulab import linalg
        inverse = hasattr(linalg, "inv")
    except ImportError:
        inverse = False
    print("# CAPABILITIES; find_displacement=%d; ulab_linalg_inv=%d" %
          (int(hasattr(img, "find_displacement")), int(inverse)))


def main():
    import sensor
    import lcd
    import time
    import gc
    import KPU as kpu

    print("# version=" + VERSION)
    task, img = None, None
    reason = "error"
    try:
        print("# stage=camera_init")
        lcd.init(freq=15000000)
        sensor.reset()
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)
        sensor.set_vflip(True)
        sensor.set_hmirror(True)
        sensor.skip_frames(time=2000)
        print("# stage=model_load")
        task = kpu.load(MODEL_ADDR)
        if kpu.init_yolo2(task, CONFIDENCE, NMS, 5, ANCHORS) is False:
            raise RuntimeError("YOLOv2 initialization failed")
        window = RGBWindow(HR_METHOD, COMPARE_METHODS)
        trace = GreenTrace()
        elapsed, frame = 0, 0
        previous_stamp = None
        last_update = -UPDATE_MS
        bpm, bpm_t_ms, state = None, -1, "WARMUP"
        print("# stage=loop_begin; window_ms=%d; bandpass=%d; sample_step=%d; method=%s; compare=%d; pos_window_ms=%d; plot_recalculate=%d; valid=0" %
              (WINDOW_MS, int(BANDPASS_ENABLED), SAMPLE_STEP, HR_METHOD,
               int(COMPARE_METHODS), POS_WINDOW_MS, int(PLOT_RECALCULATE)))
        if LOG_SAMPLES:
            print("t_ms,dt_ms,roi_x,roi_y,roi_w,roi_h,r,g,b,bpm,bpm_t_ms,samples,status,detect_ms,sample_ms,hr_ms,fps,method,pos_bpm,green_bpm,pos_state,green_state,compare_t_ms")
        while True:
            if frame == 0:
                print("# stage=first_snapshot")
            img = sensor.snapshot()
            stamp = time.ticks_ms()
            dt = 0 if previous_stamp is None else time.ticks_diff(stamp, previous_stamp)
            fps = 1000.0 / dt if dt > 0 else -1.0
            elapsed += dt  # Accumulated deltas survive the ticks_ms wraparound.
            previous_stamp = stamp
            if RUN_SECONDS > 0 and elapsed >= RUN_SECONDS * 1000:
                reason = "duration"
                break
            if frame == 0:
                report_capabilities(img)
                print("# stage=first_yolo")
            started = time.ticks_ms()
            face = choose_face(kpu.run_yolo2(task, img))
            detect_ms = time.ticks_diff(time.ticks_ms(), started)
            roi = forehead_rect(face.rect()) if face is not None else None
            if frame == 0:
                print("# stage=first_color")
            started = time.ticks_ms()
            rgb = rgb_mean(img, roi) if roi is not None else (-1.0, -1.0, -1.0)
            window.push(elapsed, rgb if roi is not None else None)
            trace.push(elapsed, rgb[1] if roi is not None else None)
            sample_ms = time.ticks_diff(time.ticks_ms(), started)
            hr_ms = 0
            if roi is None:
                bpm, bpm_t_ms, state = None, -1, "NO FACE"
                window.clear_results(state)
            if elapsed - last_update >= UPDATE_MS:
                if roi is not None:
                    started = time.ticks_ms()
                    bpm, state = window.estimate()
                    hr_ms = time.ticks_diff(time.ticks_ms(), started)
                    bpm_t_ms = elapsed if bpm is not None else -1
                last_update = elapsed
                print("# HR_RESULT; t_ms=%d; state=%s; g=%.2f; bpm=%.1f; n=%d; dt_ms=%d; detect_ms=%d; sample_ms=%d; hr_ms=%d; fps=%.1f; bpm_t_ms=%d; valid=0; method=%s; pos_bpm=%.1f; green_bpm=%.1f; pos_state=%s; green_state=%s; compare_t_ms=%d" %
                      (elapsed, state, rgb[1], -1.0 if bpm is None else bpm,
                       len(window.points), dt, detect_ms, sample_ms, hr_ms, fps, bpm_t_ms,
                       window.method, -1.0 if window.pos_bpm is None else window.pos_bpm,
                       -1.0 if window.green_bpm is None else window.green_bpm,
                       window.pos_state, window.green_state, window.compare_t_ms))
            elif roi is not None and state == "NO FACE":
                state = "WARMUP"
                window.clear_results(state)

            # All RGB reading is complete before the first overlay.
            if face is not None:
                img.draw_rectangle(face.rect(), color=(255, 0, 0), thickness=1)
            if roi is not None:
                img.draw_rectangle(roi, color=(0, 255, 0), thickness=2)
            label = "EXP %.0f BPM" % bpm if bpm is not None else "EXP -- BPM"
            img.draw_string(2, 2, label, color=(255, 255, 0), scale=2)
            # Actual loop FPS includes detection, calculation, LCD and serial
            # time between snapshots. The first frame has no measured interval.
            fps_label = "FPS:%.1f" % fps if fps >= 0 else "FPS:--"
            resolution_label = "%dx%d" % (img.width(), img.height())
            # MaixPy's default ASCII font is 8x12 at scale 1; align both rows
            # to the right edge with a 4-pixel margin and fixed-width glyphs.
            img.draw_string(img.width() - 4 - 8 * len(fps_label), 2, fps_label,
                            color=(255, 255, 255), mono_space=True)
            img.draw_string(img.width() - 4 - 8 * len(resolution_label), 16,
                            resolution_label, color=(255, 255, 255), mono_space=True)
            if bpm is None:
                img.draw_string(2, 24, state, color=(255, 255, 255))
            draw_green_trace(img, window.points, elapsed, trace.points)
            lcd.display(img)
            if LOG_SAMPLES:
                box = roi if roi is not None else (-1, -1, -1, -1)
                print("%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.1f,%d,%d,%s,%d,%d,%d,%.1f,%s,%.1f,%.1f,%s,%s,%d" %
                      ((elapsed, dt) + box + rgb +
                       (-1.0 if bpm is None else bpm, bpm_t_ms, len(window.points),
                        state, detect_ms, sample_ms, hr_ms, fps, window.method,
                        -1.0 if window.pos_bpm is None else window.pos_bpm,
                        -1.0 if window.green_bpm is None else window.green_bpm,
                        window.pos_state, window.green_state, window.compare_t_ms)))
            frame += 1
            if frame % 60 == 0:
                gc.collect()
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as exc:
        if str(exc) == "IDE interrupt":
            reason = "interrupted"
        else:
            print("# error=" + str(exc))
            raise
    finally:
        if task is not None:
            try:
                kpu.deinit(task)
            except Exception as exc:
                print("# cleanup_error=" + str(exc))
        print("# capture_stop; reason=" + reason)
        if img is not None:
            try:
                img.draw_rectangle((0, 0, 320, 44), color=(0, 0, 0), fill=True)
                img.draw_string(2, 2, "STOPPED", color=(255, 255, 255), scale=2)
                lcd.display(img)
            except Exception:
                pass


if __name__ == "__main__":
    main()

# IDE_EOF_GUARD