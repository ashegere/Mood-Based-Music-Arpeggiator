import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import imgLogo from '../assets/logo.png';
import { generateArpeggio, saveFavorite } from '../services/api';

// Navbar / dropdown icons
import imgChevron     from '../assets/449b1fcb59b5e74748df0704e11f839a4e0eb2b8.svg';
import imgUserAvatar  from '../assets/a83aa3225e8cb2ae8048839e692c745d6a1e5a8e.png';
import imgIconProfile from '../assets/f9ad290898dba738c4d95d3778299aa70d386c67.svg';
import imgIconMidi    from '../assets/7c89271e0d8c14fe49699b9d922aca42499e4e53.svg';
import imgIconSettings from '../assets/05432df0670198b2518870303f08acbab376865a.svg';
import imgIconLogout  from '../assets/b6bb2145169c29c365889f4c89ac97038f83e06c.svg';

// Instrument icons
import imgIconSynth  from '../assets/d8f5001da4ea5a2b30f8f4bdb65a88e015581042.svg';
import imgIconHarp   from '../assets/81aeef519b4cf32b2483a12daf41796b72a4c487.svg';
import imgIconPiano  from '../assets/b205a76fa03a823c4f619c891d326b1ba46322a0.svg';
import imgIconGuitar from '../assets/4e0eedadecaedf1d44b0b85b08ce65b4031b996f.svg';

// Action icons
import imgIconPlay        from '../assets/6130a43ff2d4390481756bf3ea055463480c7dc4.svg';
import imgIconDownload    from '../assets/7e755ab45c5f72177b624281f9d4d7b4fae6b60a.svg';
import imgIconChevronDown from '../assets/1407e94b406a0520ab1159ef653e16ea561da984.svg';

// ── Constants ────────────────────────────────────────────────────────────────
const KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

const SCALES = [
  { label: 'Major',           value: 'major' },
  { label: 'Natural Minor',   value: 'natural_minor' },
  { label: 'Harmonic Minor',  value: 'harmonic_minor' },
  { label: 'Dorian',          value: 'dorian' },
  { label: 'Mixolydian',      value: 'mixolydian' },
  { label: 'Pentatonic Major',value: 'pentatonic_major' },
  { label: 'Pentatonic Minor',value: 'pentatonic_minor' },
  { label: 'Blues',           value: 'blues' },
];


const INSTRUMENTS = [
  { id: 'synth',  label: 'Synth Pluck', icon: imgIconSynth,  oscType: 'sawtooth' },
  { id: 'piano',  label: 'Piano',       icon: imgIconPiano,  oscType: 'triangle' },
  { id: 'guitar', label: 'Guitar',      icon: imgIconGuitar, oscType: 'triangle' },
  { id: 'harp',   label: 'Harp',        icon: imgIconHarp,   oscType: 'sine'     },
];

const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const pitchToName = (p) => `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
const midiToFreq  = (p) => 440 * Math.pow(2, (p - 69) / 12);

// ── Helpers ──────────────────────────────────────────────────────────────────
function computeStats(notes, tempo) {
  if (!notes.length) return null;
  const pitches = notes.map(n => n.pitch);
  const avgVel  = Math.round(notes.reduce((s, n) => s + n.velocity, 0) / notes.length);
  return {
    patternLength: `${notes.length} notes`,
    noteRange:     `${pitchToName(Math.min(...pitches))} – ${pitchToName(Math.max(...pitches))}`,
    avgVelocity:   String(avgVel),
    tempo:         `${tempo} BPM`,
  };
}

function base64ToBlob(b64) {
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: 'audio/midi' });
}

// ── Visualization SVG ────────────────────────────────────────────────────────
function NoteBarChart({ notes }) {
  if (!notes.length) return null;

  const W = 560, H = 180, PAD_LEFT = 4, PAD_RIGHT = 4, PAD_TOP = 12, PAD_BOTTOM = 28;
  const chartW = W - PAD_LEFT - PAD_RIGHT;
  const chartH = H - PAD_TOP - PAD_BOTTOM;

  const pitches = notes.map(n => n.pitch);
  const minP = Math.min(...pitches), maxP = Math.max(...pitches);
  const range = maxP - minP || 1;

  // Limit to 16 bars for readability
  const display = notes.slice(0, 16);
  const barW = chartW / display.length;
  const gap  = Math.max(2, barW * 0.18);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      {display.map((note, i) => {
        const normalised = (note.pitch - minP) / range;
        const barH = Math.max(8, normalised * chartH);
        const x = PAD_LEFT + i * barW + gap / 2;
        const y = PAD_TOP + (chartH - barH);
        const label = pitchToName(note.pitch);
        return (
          <g key={i}>
            <rect
              x={x} y={y}
              width={barW - gap} height={barH}
              rx={4} ry={4}
              fill="#bfff3e"
              opacity={0.85 + normalised * 0.15}
            />
            <text
              x={x + (barW - gap) / 2}
              y={H - 6}
              textAnchor="middle"
              fontSize="10"
              fill="rgba(255,255,255,0.6)"
              fontFamily="Archivo, sans-serif"
            >{label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function VelocityLine({ notes }) {
  if (!notes.length) return null;

  const W = 560, H = 72, PAD = 16;
  const display = notes.slice(0, 16);
  const step = (W - PAD * 2) / Math.max(display.length - 1, 1);

  const points = display.map((n, i) => {
    const x = PAD + i * step;
    const y = H - PAD - ((n.velocity / 127) * (H - PAD * 2));
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <polyline
        points={points}
        fill="none"
        stroke="#bfff3e"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {display.map((n, i) => (
        <circle
          key={i}
          cx={PAD + i * step}
          cy={H - PAD - ((n.velocity / 127) * (H - PAD * 2))}
          r="3"
          fill="#bfff3e"
        />
      ))}
    </svg>
  );
}

// ── Component ────────────────────────────────────────────────────────────────
const Build = () => {
  const navigate = useNavigate();

  // Auth / UI
  const [userName, setUserName]       = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Form
  const [tempo, setTempo]               = useState(120);
  const [selectedKey, setSelectedKey]   = useState('C');
  const [selectedScale, setSelectedScale] = useState('major');
  const [numBars, setNumBars] = useState(2);
  const [numNotes, setNumNotes]         = useState(12);
  const [mood, setMood]                 = useState('');
  const [moodError, setMoodError]       = useState(false);
  const [moodShake, setMoodShake]       = useState(0);
  const [isFavorited, setIsFavorited]   = useState(false);
  const [favToast, setFavToast]         = useState(false);
  const [instrument, setInstrument]     = useState('synth');

  // Generation state
  const [generating, setGenerating] = useState(false);
  const [error, setError]           = useState('');
  const [generated, setGenerated]   = useState(false);
  const [notes, setNotes]                   = useState([]);
  const [midiBase64, setMidiBase64]         = useState('');
  const [durationSeconds, setDurationSeconds] = useState(0);
  const [stats, setStats]                   = useState(null);
  const [alignmentScore, setAlignmentScore] = useState(null);

  // Playback
  const [isPlaying, setIsPlaying]   = useState(false);
  const audioCtxRef   = useRef(null);
  const scheduledRef  = useRef([]);

  // ── Auth ────────────────────────────────────────────────────────────────
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) { navigate('/login'); return; }
    try {
      const user = JSON.parse(localStorage.getItem('user') || '{}');
      setUserName(user.full_name || user.name || user.email?.split('@')[0] || 'User');
    } catch { setUserName('User'); }
  }, [navigate]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target))
        setDropdownOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/');
  };

  // ── Generate ─────────────────────────────────────────────────────────────
  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!mood.trim()) { setMoodError(true); setMoodShake(s => s + 1); return; }
    setError('');
    setGenerating(true);
    stopPlayback();

    // 2-second loader delay before hitting the API
    await new Promise(r => setTimeout(r, 2000));

    try {
      const result = await generateArpeggio({
        key:        selectedKey,
        scale:      selectedScale,
        tempo:      tempo,
        note_count: numNotes,
        mood:       mood.trim(),
        octave:     4,
        pattern:    'ascending',
        bars:       numBars,
      });

      setNotes(result.notes || []);
      setMidiBase64(result.midi_base64 || '');
      setDurationSeconds(result.duration_seconds || 0);
      setStats(computeStats(result.notes || [], result.tempo || tempo));
      setAlignmentScore(result.alignment_score ?? null);
      setIsFavorited(false);
      setGenerated(true);
    } catch (err) {
      setError(err.message || 'Generation failed — is the backend running?');
    } finally {
      setGenerating(false);
    }
  };

  // ── Export MIDI ──────────────────────────────────────────────────────────
  const handleExport = () => {
    if (!midiBase64) return;
    const blob = base64ToBlob(midiBase64);
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `arpeggio_${selectedKey}_${selectedScale}_${tempo}bpm.mid`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Web Audio Preview ────────────────────────────────────────────────────
  const stopPlayback = () => {
    scheduledRef.current.forEach(n => { try { n.stop(0); } catch {} });
    scheduledRef.current = [];
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    setIsPlaying(false);
  };

  const handlePreview = () => {
    if (!notes.length) return;
    if (isPlaying) { stopPlayback(); return; }

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = ctx;
    const beatDur = 60 / tempo;
    const oscType = INSTRUMENTS.find(i => i.id === instrument)?.oscType || 'sawtooth';
    const oscNodes = [];

    let lastEnd = 0;
    notes.forEach(note => {
      const startSec = ctx.currentTime + note.position * beatDur;
      const durSec   = Math.max(0.05, note.duration * beatDur);
      const gain     = (note.velocity / 127) * 0.25;
      lastEnd        = Math.max(lastEnd, startSec + durSec);

      const osc  = ctx.createOscillator();
      const amp  = ctx.createGain();

      osc.type = oscType;
      osc.frequency.setValueAtTime(midiToFreq(note.pitch), startSec);

      amp.gain.setValueAtTime(0, startSec);
      amp.gain.linearRampToValueAtTime(gain, startSec + 0.01);
      amp.gain.exponentialRampToValueAtTime(0.0001, startSec + durSec * 0.85);

      osc.connect(amp);
      amp.connect(ctx.destination);
      osc.start(startSec);
      osc.stop(startSec + durSec);
      oscNodes.push(osc);
    });

    scheduledRef.current = oscNodes;
    setIsPlaying(true);

    const totalMs = (lastEnd - ctx.currentTime + 0.3) * 1000;
    setTimeout(() => {
      ctx.close().catch(() => {});
      audioCtxRef.current = null;
      setIsPlaying(false);
    }, totalMs);
  };

  // cleanup on unmount
  useEffect(() => () => stopPlayback(), []);

  const firstName = userName.split(' ')[0];

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="build-root">
      <style>{`
        @import url('https://api.fontshare.com/v2/css?f[]=clash-grotesk@200,300,400,500,600,700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        .build-root {
          font-family: 'ClashGrotesk-Variable', 'Clash Grotesk', sans-serif;
          min-height: 100vh;
          background: #0000ff;
          display: flex;
          flex-direction: column;
          padding: 11px 26px 19px;
          gap: 19px;
          overflow-x: hidden;
        }

        /* ── NAVBAR ── */
        .build-navbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: 100%;
          position: relative;
          height: 64px;
          flex-shrink: 0;
          z-index: 200;
          animation: fadeInDown 0.6s cubic-bezier(0.22,1,0.36,1) both;
        }

        .build-logo {
          display: flex;
          align-items: center;
          cursor: pointer;
          text-decoration: none;
        }
        .build-logo img { height: 36px; width: auto; object-fit: contain; display: block; }

        .build-nav-links {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 40px;
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(245,245,245,0.2);
          border-radius: 40px;
          padding: 13px 28px;
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          white-space: nowrap;
        }
        .build-nav-link { font-size: 15px; font-weight: 500; color: #ffffff; text-decoration: none; transition: opacity 0.15s; }
        .build-nav-link:hover { opacity: 0.75; }

        /* ── PROFILE DROPDOWN ── */
        .build-user-wrap { position: relative; flex-shrink: 0; }

        .build-user-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 50%;
          width: 44px;
          height: 44px;
          cursor: pointer;
          color: #ffffff;
          transition: background 0.18s, border-color 0.18s, transform 0.18s;
        }
        .build-user-btn:hover {
          background: rgba(255,255,255,0.22);
          border-color: rgba(255,255,255,0.5);
          transform: scale(1.08);
        }

        .build-dropdown {
          position: absolute;
          top: calc(100% + 8px);
          right: 0;
          min-width: 240px;
          background: #0a0aee;
          border: 1.5px solid #ffffff;
          border-radius: 10px;
          overflow: hidden;
          z-index: 100;
          box-shadow: 0 4px 16px rgba(0,0,0,0.15);
          animation: dropdownIn 0.18s cubic-bezier(0.22,1,0.36,1) both;
          padding: 5px 0 10px;
        }

        @keyframes dropdownIn {
          from { opacity: 0; transform: translateY(-8px) scale(0.97); }
          to   { opacity: 1; transform: translateY(0)   scale(1); }
        }

        .build-dropdown-header {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          padding: 20px 19px 16px;
          border-bottom: 1px solid rgba(255,255,255,0.15);
        }
        .build-dropdown-avatar {
          width: 34px; height: 34px;
          border-radius: 50%;
          flex-shrink: 0;
          border: 1.5px solid rgba(255,255,255,0.4);
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          background: rgba(255,255,255,0.1);
        }
        .build-dropdown-hi { font-size: 14px; font-weight: 600; color: #ffffff; letter-spacing: -0.2px; }

        .build-dropdown-item {
          display: flex;
          align-items: center;
          gap: 13px;
          padding: 12px 19px;
          font-size: 13px;
          font-weight: 500;
          font-family: 'Archivo', sans-serif;
          color: #ffffff;
          text-decoration: none;
          cursor: pointer;
          background: none;
          border: none;
          width: 100%;
          text-align: left;
          transition: background 0.12s;
        }
        .build-dropdown-item:hover { background: rgba(255,255,255,0.1); }
        .build-dropdown-item.logout { color: rgba(255,255,255,0.75); }
        .build-dropdown-item.logout:hover { background: rgba(255,80,80,0.12); color: #ff8080; }
        .build-dropdown-icon {
          width: 14px;
          height: 14px;
          min-width: 14px;
          min-height: 14px;
          flex-shrink: 0;
          opacity: 0.85;
          display: block;
          object-fit: contain;
        }
        img.build-dropdown-icon { filter: brightness(0) invert(1); }
        .build-dropdown-divider { height: 1px; background: rgba(255,255,255,0.12); margin: 5px 0; }

        /* ── MAIN ── */
        .build-main {
          display: flex;
          gap: 24px;
          align-items: stretch;
          flex: 1;
          max-width: 1200px;
          width: 100%;
          margin: 0 auto;
          animation: fadeInDown 0.7s cubic-bezier(0.22,1,0.36,1) 0.1s both;
        }

        /* ── CARDS ── */
        .build-card {
          background: #0000ff;
          border: 2.5px solid #ffffff;
          border-radius: 16px;
          box-shadow: 0 4px 4px rgba(0,0,0,0.25);
          padding: 36px 36px;
          flex: 1;
          min-width: 0;
          display: flex;
          flex-direction: column;
          gap: 22px;
        }
        .build-card-title { font-size: clamp(20px, 2.4vw, 30px); font-weight: 600; color: #ffffff; letter-spacing: -0.9px; line-height: 1; }

        /* ── FORM ── */
        .build-form { display: flex; flex-direction: column; gap: 18px; flex: 1; }
        .build-field { display: flex; flex-direction: column; gap: 8px; }
        .build-label { font-size: 13px; font-weight: 500; color: #ffffff; font-family: 'Archivo', sans-serif; }

        /* Slider */
        .build-slider-row { display: flex; align-items: center; gap: 14px; }
        .build-slider {
          flex: 1; -webkit-appearance: none; appearance: none;
          height: 7px; border-radius: 10px;
          background: rgba(255,255,255,0.2); outline: none; cursor: pointer;
        }
        .build-slider::-webkit-slider-thumb {
          -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%;
          background: #bfff3e; cursor: pointer; border: 2px solid #ffffff;
          box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }
        .build-slider::-moz-range-thumb {
          width: 20px; height: 20px; border-radius: 50%;
          background: #bfff3e; cursor: pointer; border: 2px solid #ffffff;
        }
        .build-slider-value { font-size: 15px; font-weight: 500; color: #bfff3e; min-width: 34px; text-align: right; flex-shrink: 0; }

        /* Key grid */
        .build-key-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
        .build-key-btn {
          padding: 9px 4px; border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.2);
          background: rgba(255,255,255,0.1);
          color: #ffffff; font-size: 13px; font-weight: 500; font-family: 'Archivo', sans-serif;
          text-align: center; cursor: pointer;
          transition: background 0.12s, color 0.12s, border-color 0.12s;
        }
        .build-key-btn.selected { background: #ffffff; color: #0000ff; border-color: #ffffff; }
        .build-key-btn:hover:not(.selected) { background: rgba(255,255,255,0.2); }


        /* Dropdown select */
        .build-select {
          width: 100%; height: 44px;
          background: rgba(255,255,255,0.1);
          border: 1.5px solid rgba(255,255,255,0.2);
          border-radius: 8px; padding: 0 16px;
          font-size: 13px; font-weight: 500; font-family: 'Archivo', sans-serif;
          color: #ffffff; outline: none; cursor: pointer;
          appearance: none; -webkit-appearance: none;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='white' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
          background-repeat: no-repeat; background-position: right 18px center;
          transition: border-color 0.15s;
        }
        .build-select:focus { border-color: rgba(255,255,255,0.5); }
        .build-select option { background: #0000ff; color: #ffffff; }

        /* Textarea */
        .build-textarea {
          width: 100%; min-height: 90px;
          background: rgba(255,255,255,0.1);
          border: 1.5px solid rgba(255,255,255,0.2);
          border-radius: 8px; padding: 12px 16px;
          font-size: 13px; font-weight: 500; font-family: 'Archivo', sans-serif;
          color: #ffffff; outline: none; resize: vertical;
          transition: border-color 0.15s;
        }
        .build-textarea::placeholder { color: rgba(255,255,255,0.5); }
        .build-textarea:focus { border-color: rgba(255,255,255,0.5); }
        .build-textarea.error { border-color: #ff3b3b; animation: fieldShake 0.4s cubic-bezier(0.36,0.07,0.19,0.97) both; }
        .build-textarea.error:focus { border-color: #ff3b3b; }
        @keyframes fieldShake {
          0%,100% { transform: translateX(0); }
          15%     { transform: translateX(-7px); }
          30%     { transform: translateX(7px); }
          45%     { transform: translateX(-5px); }
          60%     { transform: translateX(5px); }
          75%     { transform: translateX(-2px); }
          90%     { transform: translateX(2px); }
        }

        /* Mood error tooltip */
        .build-mood-error-wrap { position: relative; }
        .build-mood-error {
          position: absolute;
          top: 8px;
          left: 0;
          z-index: 10;
          display: inline-flex;
          align-items: center;
          gap: 7px;
          background: #ffffff;
          border-radius: 8px;
          padding: 8px 14px;
          width: fit-content;
          animation: toastIn 0.2s cubic-bezier(0.22,1,0.36,1) both;
        }
        @keyframes toastIn {
          from { opacity: 0; transform: translateY(-4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .build-mood-error::before {
          content: '';
          position: absolute;
          top: -7px;
          left: 18px;
          width: 0; height: 0;
          border-left: 7px solid transparent;
          border-right: 7px solid transparent;
          border-bottom: 7px solid #ffffff;
        }
        .build-mood-error-icon {
          color: #ff3b3b;
          font-size: 14px;
          flex-shrink: 0;
          line-height: 1;
        }
        .build-mood-error-text {
          font-size: 13px;
          font-weight: 500;
          color: #ff3b3b;
          font-family: 'Archivo', sans-serif;
          white-space: nowrap;
        }

        /* Error */
        .build-error {
          background: rgba(255,80,80,0.15);
          border: 1px solid rgba(255,80,80,0.4);
          border-radius: 10px;
          padding: 8px 13px;
          font-size: 11px;
          font-family: 'Archivo', sans-serif;
          color: #ff8080;
          text-align: center;
        }

        /* Generate button */
        .build-generate-btn {
          width: 100%; height: 52px;
          background: #bfff3e; border: none; border-radius: 50px;
          font-size: 16px; font-weight: 500; font-family: inherit;
          color: #060511; cursor: pointer;
          transition: opacity 0.15s, transform 0.15s;
          margin-top: 4px; letter-spacing: -0.3px;
        }
        .build-generate-btn:hover { opacity: 0.9; transform: translateY(-1px); }
        .build-generate-btn:disabled { opacity: 0.65; cursor: not-allowed; transform: none; }

        /* ── LOADER ── */
        .build-loader {
          width: 100%;
          height: 52px;
          background: #bfff3e;
          border-radius: 50px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          margin-top: 4px;
        }

        .build-loader-note {
          font-size: 18px;
          color: #060511;
          animation: notesBounce 0.6s ease-in-out infinite alternate;
        }
        .build-loader-note:nth-child(1) { animation-delay: 0s; }
        .build-loader-note:nth-child(2) { animation-delay: 0.15s; }
        .build-loader-note:nth-child(3) { animation-delay: 0.3s; }

        @keyframes notesBounce {
          from { transform: translateY(0); }
          to   { transform: translateY(-10px); }
        }

        /* ── RIGHT CARD ── */
        .build-preview-card { display: flex; flex-direction: column; gap: 16px; }

        .build-viz-panel {
          background: rgba(255,255,255,0.08);
          border: 1.5px solid rgba(255,255,255,0.1);
          border-radius: 13px;
          flex: 1; min-height: 0;
          display: flex; flex-direction: column;
          overflow: hidden;
        }

        /* Empty state */
        .build-viz-empty {
          flex: 1; display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          gap: 10px; padding: 26px 19px;
        }
        .build-viz-empty-icon { width: 45px; height: 45px; opacity: 0.3; }
        .build-viz-empty-text { font-size: 13px; font-weight: 500; font-family: 'Archivo', sans-serif; color: rgba(255,255,255,0.5); text-align: center; }

        /* Full state */
        .build-viz-charts {
          display: flex; flex-direction: column;
          flex: 1; min-height: 0;
          padding: 12px 14px 0;
          gap: 0;
          animation: vizReveal 0.5s cubic-bezier(0.22,1,0.36,1) both;
        }
        .build-viz-barchart-wrap {
          flex: 1; min-height: 0; overflow: hidden;
        }
        .build-viz-barchart-wrap svg {
          width: 100%; height: 100%;
        }
        @keyframes vizReveal {
          0%   { opacity: 0; transform: translateY(10px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .build-viz-charts > * {
          animation: vizReveal 0.5s cubic-bezier(0.22,1,0.36,1) both;
        }
        .build-viz-charts > *:nth-child(1) { animation-delay: 0.05s; }
        .build-viz-charts > *:nth-child(2) { animation-delay: 0.12s; }
        .build-viz-charts > *:nth-child(3) { animation-delay: 0.19s; }

        .build-viz-divider {
          height: 1px;
          background: rgba(255,255,255,0.1);
          margin: 10px 0;
        }
        .build-viz-velocity-wrap {
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
          overflow: hidden;
          margin-bottom: 12px;
        }

        .build-viz-stats {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          border-top: 1px solid rgba(255,255,255,0.1);
          margin: 0 -14px -1px;
        }
        .build-viz-stat {
          padding: 11px 13px;
          display: flex; flex-direction: column; gap: 4px;
          border-right: 1px solid rgba(255,255,255,0.1);
          margin-bottom: 0;
        }
        .build-viz-stat:last-child { border-right: none; }
        .build-viz-stat-label { font-size: 9px; font-weight: 500; font-family: 'Archivo', sans-serif; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px; }
        .build-viz-stat-value { font-size: 12px; font-weight: 600; font-family: 'Archivo', sans-serif; color: #ffffff; white-space: nowrap; }

        /* Alignment badge */
        .build-alignment {
          display: inline-flex; align-items: center; gap: 5px;
          background: rgba(191,255,62,0.15);
          border: 1px solid rgba(191,255,62,0.4);
          border-radius: 16px; padding: 3px 10px;
          font-size: 10px; font-weight: 500; font-family: 'Archivo', sans-serif;
          color: #bfff3e; align-self: flex-start;
        }

        /* Instrument mode */
        .build-instrument-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
        .build-instrument-btn {
          display: flex; align-items: center; justify-content: center;
          gap: 8px; padding: 12px 10px; border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.2);
          background: rgba(255,255,255,0.1);
          color: #ffffff; font-size: 13px; font-weight: 500; font-family: 'Archivo', sans-serif;
          cursor: pointer; transition: background 0.12s, color 0.12s, border-color 0.12s;
        }
        .build-instrument-btn.selected { background: #ffffff; color: #0000ff; border-color: #ffffff; }
        .build-instrument-btn:hover:not(.selected) { background: rgba(255,255,255,0.2); }
        .build-instrument-icon { width: 14px; height: 14px; flex-shrink: 0; filter: brightness(0) invert(1); }
        .build-instrument-btn.selected .build-instrument-icon { filter: brightness(0); }

        /* Action buttons */
        .build-actions { display: flex; gap: 13px; margin-top: auto; padding-top: 4px; }

        .build-preview-btn {
          flex: 1; height: 52px;
          background: #ffffff; border: 2.5px solid #ffffff; border-radius: 50px;
          display: flex; align-items: center; justify-content: center; gap: 8px;
          font-size: 16px; font-weight: 500; font-family: inherit;
          color: #0000ff; cursor: pointer; transition: opacity 0.15s;
        }
        .build-preview-btn:hover { opacity: 0.88; }
        .build-preview-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .build-preview-btn.playing { background: #bfff3e; border-color: #bfff3e; color: #060511; }

        .build-export-btn {
          flex: 1; height: 52px;
          background: #d9ff00; border: none; border-radius: 50px;
          display: flex; align-items: center; justify-content: center; gap: 8px;
          font-size: 16px; font-weight: 500; font-family: inherit;
          color: #03061f; cursor: pointer; transition: opacity 0.15s;
        }
        .build-export-btn:hover { opacity: 0.9; }
        .build-export-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        .build-action-icon { width: 14px; height: 14px; flex-shrink: 0; }

        .build-fav-btn {
          width: 52px; height: 52px; flex-shrink: 0;
          background: transparent;
          border: 2px solid rgba(255,255,255,0.3);
          border-radius: 50%;
          display: flex; align-items: center; justify-content: center;
          cursor: pointer;
          color: #ffffff;
          transition: border-color 0.18s, background 0.18s, color 0.18s, transform 0.18s;
        }
        .build-fav-btn:hover {
          border-color: #ffffff;
          background: rgba(255,255,255,0.08);
          transform: scale(1.08);
        }
        .build-fav-btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
        .build-fav-btn.saved {
          background: rgba(255,255,255,0.12);
          border-color: #ffffff;
          color: #bfff3e;
        }

        .build-fav-toast {
          position: fixed;
          top: 28px;
          left: 50%;
          z-index: 9999;
          display: flex; align-items: center; gap: 10px;
          background: #ffffff;
          border-radius: 12px;
          padding: 13px 20px;
          font-size: 14px; font-weight: 500;
          font-family: 'Archivo', sans-serif;
          color: #0000ff;
          box-shadow: 0 8px 24px rgba(0,0,0,0.18);
          white-space: nowrap;
          animation: toastLifecycle 2.5s cubic-bezier(0.22,1,0.36,1) forwards;
        }
        @keyframes toastLifecycle {
          0%   { opacity: 0; transform: translateX(-50%) translateY(-12px); }
          12%  { opacity: 1; transform: translateX(-50%) translateY(0); }
          75%  { opacity: 1; transform: translateX(-50%) translateY(0); }
          100% { opacity: 0; transform: translateX(-50%) translateY(-8px); }
        }

        /* ── FOOTER ── */
        .build-footer {
          display: flex; align-items: center; justify-content: space-between;
          padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.2);
          flex-shrink: 0;
          max-width: 1200px;
          width: 100%;
          margin: 0 auto;
          animation: fadeInDown 0.5s cubic-bezier(0.22,1,0.36,1) 0.3s both;
        }
        .build-footer-copy { font-size: 11px; font-weight: 500; font-family: 'Archivo', sans-serif; color: rgba(255,255,255,0.6); }
        .build-footer-coffee {
          background: rgba(255,255,255,0.1); border: none; border-radius: 8px;
          padding: 6px 16px; height: 32px; font-size: 11px; font-weight: 500;
          font-family: inherit; color: #ffffff; cursor: pointer;
          text-decoration: none; display: flex; align-items: center;
          transition: opacity 0.15s; white-space: nowrap;
        }
        .build-footer-coffee:hover { opacity: 0.75; }

        /* ── ANIMATIONS ── */
        @keyframes fadeInDown {
          from { opacity: 0; transform: translateY(-20px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        /* ── RESPONSIVE ── */
        @media (max-width: 1024px) {
          .build-root { padding: 13px 16px; gap: 13px; }
          .build-main { flex-direction: column; }
          .build-card { padding: 19px 19px; }
          .build-viz-stats { grid-template-columns: repeat(2, 1fr); }
          .build-viz-stat:nth-child(2) { border-right: none; }
          .build-viz-stat:nth-child(3) { border-right: 1px solid rgba(255,255,255,0.1); }
        }
        @media (max-width: 768px) {
          .build-root { padding: 10px 13px; gap: 10px; }
          .build-nav-links { display: none; }
          .build-card { padding: 16px 13px; gap: 13px; }
          .build-card-title { font-size: 18px; }
          .build-key-btn { font-size: 10px; padding: 5px 2px; }
          .build-generate-btn { height: 37px; font-size: 13px; }
          .build-preview-btn, .build-export-btn { height: 37px; font-size: 12px; }
          .build-footer { flex-direction: column; gap: 10px; align-items: flex-start; }
        }
        @media (max-width: 480px) {
          .build-root { padding: 12px; }
          .build-key-grid { grid-template-columns: repeat(4, 1fr); }
          .build-actions { flex-direction: column; }
        }
      `}</style>

      {/* ── NAVBAR ── */}
      <nav className="build-navbar">
        <Link to="/" className="build-logo" aria-label="Arpeggiate home">
          <img src={imgLogo} alt="Arpeggiate" />
        </Link>

        <div className="build-nav-links">
          <Link to="/build" className="build-nav-link">Home</Link>
          <a href="/#create" className="build-nav-link">Explore</a>
          <a href="/#create" className="build-nav-link">Create</a>
        </div>

        <div className="build-user-wrap" ref={dropdownRef}>
          <button
            className="build-user-btn"
            onClick={() => setDropdownOpen(o => !o)}
            aria-expanded={dropdownOpen}
            aria-label="Profile menu"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
            </svg>
          </button>

          {dropdownOpen && (
            <div className="build-dropdown">
              <div className="build-dropdown-header">
                <div className="build-dropdown-avatar">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                  </svg>
                </div>
                <span className="build-dropdown-hi">Hi, {firstName}</span>
              </div>
              <button className="build-dropdown-item">
                <svg className="build-dropdown-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                </svg>
                View Profile
              </button>
              <button className="build-dropdown-item" onClick={() => navigate('/saved')}>
                <img src={imgIconMidi} alt="" className="build-dropdown-icon" />
                Saved MIDIs
              </button>
              <button className="build-dropdown-item">
                <svg className="build-dropdown-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="5" width="20" height="14" rx="2"/>
                  <line x1="2" y1="10" x2="22" y2="10"/>
                  <line x1="6" y1="15" x2="10" y2="15"/>
                </svg>
                Billing
              </button>
              <button className="build-dropdown-item">
                <img src={imgIconSettings} alt="" className="build-dropdown-icon" />
                Settings
              </button>
              <div className="build-dropdown-divider" />
              <button className="build-dropdown-item logout" onClick={handleLogout}>
                <img src={imgIconLogout} alt="" className="build-dropdown-icon" />
                Log out
              </button>
            </div>
          )}
        </div>
      </nav>

      {/* ── MAIN ── */}
      <div className="build-main">

        {/* LEFT — Build */}
        <div className="build-card">
          <h1 className="build-card-title">Build Your Arpeggio</h1>

          <form className="build-form" onSubmit={handleGenerate}>

            <div className="build-field">
              <label className="build-label">Tempo (BPM)</label>
              <div className="build-slider-row">
                <input type="range" className="build-slider" min={40} max={240} step={1}
                  value={tempo} onChange={e => setTempo(Number(e.target.value))} />
                <span className="build-slider-value">{tempo}</span>
              </div>
            </div>

            <div className="build-field">
              <label className="build-label">Key</label>
              <div className="build-key-grid">
                {KEYS.map(k => (
                  <button key={k} type="button"
                    className={`build-key-btn${selectedKey === k ? ' selected' : ''}`}
                    onClick={() => setSelectedKey(k)}>{k}</button>
                ))}
              </div>
            </div>

            <div className="build-field">
              <label className="build-label">Scale</label>
              <select className="build-select" value={selectedScale}
                onChange={e => setSelectedScale(e.target.value)}>
                {SCALES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </div>

            <div className="build-field">
              <label className="build-label">Number of Bars</label>
              <div className="build-slider-row">
                <input type="range" className="build-slider" min={1} max={8} step={1}
                  value={numBars} onChange={e => setNumBars(Number(e.target.value))} />
                <span className="build-slider-value">{numBars}</span>
              </div>
            </div>

            <div className="build-field">
              <label className="build-label">Number of Notes</label>
              <div className="build-slider-row">
                <input type="range" className="build-slider" min={4} max={24} step={1}
                  value={numNotes} onChange={e => setNumNotes(Number(e.target.value))} />
                <span className="build-slider-value">{numNotes}</span>
              </div>
            </div>

            <div className="build-field" style={{ flex: 1 }}>
              <label className="build-label">Mood-Based Keyword/Phrase</label>
              <textarea
                key={moodShake}
                className={`build-textarea${moodError ? ' error' : ''}`}
                placeholder="e.g., happy and energetic, calm and peaceful..."
                value={mood}
                onChange={e => { setMood(e.target.value); setMoodError(false); setError(''); }}
              />
              <div className="build-mood-error-wrap">
                {moodError && (
                  <div className="build-mood-error">
                    <span className="build-mood-error-icon">⚠</span>
                    <span className="build-mood-error-text">Please fill this field</span>
                  </div>
                )}
              </div>
            </div>

            {error && <div className="build-error">{error}</div>}

            {generating ? (
              <div className="build-loader">
                <span className="build-loader-note">♫</span>
                <span className="build-loader-note">♫</span>
                <span className="build-loader-note">♫</span>
              </div>
            ) : (
              <button type="submit" className="build-generate-btn">
                Generate Arpeggio
              </button>
            )}
          </form>
        </div>

        {/* RIGHT — Preview */}
        <div className="build-card build-preview-card">
          <h2 className="build-card-title">Export and Preview</h2>

          <div className="build-field" style={{ flex: 1 }}>
            <label className="build-label">Pattern Sequence Visualization</label>
            <div className="build-viz-panel">
              {generated && notes.length ? (
                <div className="build-viz-charts" key={midiBase64}>
                  <div className="build-viz-barchart-wrap">
                    <NoteBarChart notes={notes} />
                  </div>
                  <div className="build-viz-divider" />
                  <div className="build-viz-velocity-wrap">
                    <VelocityLine notes={notes} />
                  </div>
                  <div className="build-viz-stats">
                    <div className="build-viz-stat">
                      <span className="build-viz-stat-label">Length</span>
                      <span className="build-viz-stat-value">{stats?.patternLength}</span>
                    </div>
                    <div className="build-viz-stat">
                      <span className="build-viz-stat-label">Note Range</span>
                      <span className="build-viz-stat-value">{stats?.noteRange}</span>
                    </div>
                    <div className="build-viz-stat">
                      <span className="build-viz-stat-label">Avg Velocity</span>
                      <span className="build-viz-stat-value">{stats?.avgVelocity}</span>
                    </div>
                    <div className="build-viz-stat">
                      <span className="build-viz-stat-label">Tempo</span>
                      <span className="build-viz-stat-value">{stats?.tempo}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="build-viz-empty">
                  <img src={imgIconSynth} alt="" className="build-viz-empty-icon" />
                  <p className="build-viz-empty-text">Generate a pattern to see the preview</p>
                </div>
              )}
            </div>
            {alignmentScore !== null && (
              <div className="build-alignment">
                ✦ Mood alignment: {Math.round(alignmentScore * 100)}%
              </div>
            )}
          </div>

          <div className="build-field">
            <label className="build-label">Instrument Mode</label>
            <div className="build-instrument-grid">
              {INSTRUMENTS.map(ins => (
                <button key={ins.id} type="button"
                  className={`build-instrument-btn${instrument === ins.id ? ' selected' : ''}`}
                  onClick={() => setInstrument(ins.id)}>
                  <img src={ins.icon} alt="" className="build-instrument-icon" />
                  {ins.label}
                </button>
              ))}
            </div>
          </div>

          <div className="build-actions">
            <button
              className={`build-preview-btn${isPlaying ? ' playing' : ''}`}
              onClick={handlePreview}
              disabled={!generated || !notes.length}
            >
              <img src={imgIconPlay} alt="" className="build-action-icon" />
              {isPlaying ? 'Stop' : 'Preview'}
            </button>
            <button className="build-export-btn" onClick={handleExport} disabled={!generated || !midiBase64}>
              <img src={imgIconDownload} alt="" className="build-action-icon" />
              Export
              <img src={imgIconChevronDown} alt="" className="build-action-icon" />
            </button>
            <button
              className={`build-fav-btn${isFavorited ? ' saved' : ''}`}
              onClick={async () => {
                if (isFavorited) return;
                try {
                  await saveFavorite({
                    midi_base64: midiBase64,
                    mood: mood.trim(),
                    key: selectedKey,
                    scale: selectedScale,
                    tempo,
                    note_count: notes.length,
                    duration_seconds: durationSeconds,
                  });
                  setIsFavorited(true);
                  setFavToast(true);
                  setTimeout(() => setFavToast(false), 2500);
                } catch {
                  setError('Failed to save to favorites. Please try again.');
                }
              }}
              disabled={!generated || !notes.length}
              aria-label={isFavorited ? 'Saved to favorites' : 'Save to favorites'}
              title={isFavorited ? 'Saved to favorites' : 'Save to favorites'}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill={isFavorited ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
              </svg>
            </button>
          </div>

          {favToast && (
            <div className="build-fav-toast">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
              </svg>
              MIDI saved to favorites
            </div>
          )}
        </div>
      </div>

      {/* ── FOOTER ── */}
      <footer className="build-footer">
        <span className="build-footer-copy">© 2025 Arpeggiate.ai - Powered by AI</span>
        <a href="https://buymeacoffee.com" target="_blank" rel="noopener noreferrer"
          className="build-footer-coffee">Buy me a coffee</a>
      </footer>
    </div>
  );
};

export default Build;
