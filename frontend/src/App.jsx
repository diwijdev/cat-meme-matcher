import { useEffect, useMemo, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { extractFeatures } from "./lib/features";
import { matchFace } from "./lib/api";

const MEME_FILE = {
  shocked: "OMG-Cat.jpg",
  judging: "suscat.png",
  laughing: "laughing-cat.jpg",
  polite: "polite cat.jpg",
  scared: "Scared-Cat.jpg",
  smiling: "Smiling-Cat.jpg",
  staring: "staring-cat.jpg",
  angry: "angrycat.png",
};

function MemePreview({ label }) {
  const src = useMemo(() => {
    const file = MEME_FILE[label] || MEME_FILE.staring;
    return `/memes/${file}`;
  }, [label]);

  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5">
      <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent" />
      <div className="p-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-white/70">Matched Meme</p>
          <span className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs text-white/80">
            {label}
          </span>
        </div>

        <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-black/30">
          <img src={src} alt={label} className="h-[320px] w-full object-contain" />
        </div>
      </div>
    </div>
  );
}

function ConfidenceBar({ conf }) {
  const pct = Math.round(Math.max(0, Math.min(1, conf)) * 100);
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-white/70">Confidence</p>
        <p className="text-sm font-semibold text-white">{pct}%</p>
      </div>
      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-white/70 transition-all duration-200"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-white/50">Higher means the classifier is more certain.</p>
    </div>
  );
}

function Badge({ children, tone = "neutral" }) {
  const styles =
    tone === "good"
      ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-200"
      : tone === "warn"
      ? "border-amber-400/20 bg-amber-400/10 text-amber-200"
      : "border-white/10 bg-white/10 text-white/80";

  return <span className={`rounded-full border px-3 py-1 text-xs ${styles}`}>{children}</span>;
}

export default function App() {
  const videoRef = useRef(null);
  const featsRef = useRef(null);

  const [landmarker, setLandmarker] = useState(null);

  const [label, setLabel] = useState("staring");
  const [conf, setConf] = useState(0.0);

  const [baseBrow, setBaseBrow] = useState(null);
  const browSamplesRef = useRef([]);

  useEffect(() => {
    async function init() {
      // 1) Start webcam
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      // 2) Load MediaPipe (WASM + model)
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      const lm = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      });

      setLandmarker(lm);
    }

    init();
  }, []);

  useEffect(() => {
    if (!landmarker) return;

    let rafId;

    const loop = async () => {
      const video = videoRef.current;
      const now = performance.now();

      const res = landmarker.detectForVideo(video, now);

      if (res.faceLandmarks && res.faceLandmarks.length > 0) {
        const feats = extractFeatures(res.faceLandmarks[0]);
        featsRef.current = feats;


        // Calibrate brows for ~1 sec
        if (baseBrow == null) {
          browSamplesRef.current.push(feats.brow_raise);
          if (browSamplesRef.current.length >= 30) {
            const avg =
              browSamplesRef.current.reduce((a, b) => a + b, 0) / browSamplesRef.current.length;
            setBaseBrow(avg);
          }
        }

        const payload = { ...feats, base_brow: baseBrow };

        try {
          const out = await matchFace(payload);
          setLabel(out.label);
          setConf(out.conf);
        } catch (e) {
          // Backend down or CORS issue: fall back
          setLabel("staring");
          setConf(0.0);
        }
      } else {
        setLabel("staring");
        setConf(0.0);
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [landmarker, baseBrow]);

  return (
    <div className="min-h-screen bg-[#0b0f19] text-white">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-gradient-to-b from-white/10 to-transparent" />

      <div className="relative mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Cat Meme Matcher</h1>
            <p className="mt-1 text-sm text-white/60">
              Real-time face tracking → expression features → meme match.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="good">MediaPipe Face Landmarker</Badge>
            <Badge>FastAPI</Badge>
            <Badge>React + Tailwind</Badge>
          </div>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5">
              <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent" />

              <div className="flex items-center justify-between px-5 py-4">
                <div className="flex items-center gap-3">
                  <div className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_18px_rgba(52,211,153,0.6)]" />
                  <p className="text-sm text-white/80">Live Camera</p>
                </div>

                {baseBrow == null ? <Badge tone="warn">Calibrating brows…</Badge> : <Badge tone="good">Calibrated</Badge>}
              </div>

              <div className="px-5 pb-5">
                <div className="overflow-hidden rounded-xl border border-white/10 bg-black/40">
                  <video
                    ref={videoRef}
                    className="h-[360px] w-full object-cover"
                    playsInline
                    muted
                  />
                </div>

                <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2">
                    <p className="text-sm text-white/60">Match:</p>
                    <p className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-sm font-semibold">
                      {label}
                    </p>
                  </div>

                  <div className="mt-3 text-xs text-white/60 space-y-1">
                    <div>mouth_w: {featsRef.current?.mouth_w?.toFixed(3) ?? "..."}</div>
                    <div>smile_up: {featsRef.current?.smile_up?.toFixed(3) ?? "..."}</div>
                  </div>


                  <div className="flex items-center gap-2 text-xs text-white/50">
                    {baseBrow != null && (
                      <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                        BASE_BROW: <span className="text-white/80">{baseBrow.toFixed(3)}</span>
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <MemePreview label={label} />
            <ConfidenceBar conf={conf} />
          </div>
        </div>

        <div className="mt-10 text-xs text-white/40">
          Built by Diwij Dev · React + Tailwind · MediaPipe · FastAPI
        </div>
      </div>
    </div>
  );
}
