import { useEffect, useRef } from 'react';

/**
 * The waveform doubles as the edit map: regions the transcript no longer keeps
 * are painted as cuts, so it is obvious at a glance what leaving the editor
 * would produce.
 */
export default function Waveform({ envelope, duration, currentTime, cuts, onSeek }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(width * ratio));
    canvas.height = Math.max(1, Math.floor(height * ratio));

    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const rms = envelope?.rms ?? [];
    const middle = height / 2;

    if (rms.length === 0 || !duration) {
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.fillRect(0, middle - 1, width, 2);
      return;
    }

    // Normalise against the loudest frame so quiet recordings still read.
    let peak = 0;
    for (const v of rms) if (v > peak) peak = v;
    const scale = peak > 0 ? (height / 2 - 2) / peak : 0;

    const isCut = (time) => cuts.some((c) => time >= c.start && time < c.end);

    for (let x = 0; x < width; x++) {
      const index = Math.min(rms.length - 1, Math.floor((x / width) * rms.length));
      const amplitude = Math.max(1, rms[index] * scale);
      const time = (x / width) * duration;
      ctx.fillStyle = isCut(time) ? 'rgba(248,113,113,0.35)' : 'rgba(125,211,252,0.75)';
      ctx.fillRect(x, middle - amplitude, 1, amplitude * 2);
    }

    if (currentTime >= 0 && duration > 0) {
      const x = (currentTime / duration) * width;
      ctx.fillStyle = '#fbbf24';
      ctx.fillRect(x - 1, 0, 2, height);
    }
  }, [envelope, duration, currentTime, cuts]);

  return (
    <canvas
      ref={canvasRef}
      className="waveform"
      title="Click to seek"
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const fraction = (event.clientX - rect.left) / rect.width;
        onSeek(Math.max(0, Math.min(1, fraction)) * duration);
      }}
    />
  );
}
