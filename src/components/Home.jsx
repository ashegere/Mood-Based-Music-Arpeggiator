import React, { useState, useEffect, useRef } from 'react';
import { Music, LogOut, Sparkles, Play, Square, Download, ChevronDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { generateArpeggio } from '../services/api';
import './Home.css';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const pitchToName = (p) => `${PITCH_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
const midiToFreq  = (p) => 440 * Math.pow(2, (p - 69) / 12);

const SCALES = [
  { value: 'major',           label: 'Major' },
  { value: 'minor',           label: 'Minor' },
  { value: 'dorian',          label: 'Dorian' },
  { value: 'phrygian',        label: 'Phrygian' },
  { value: 'lydian',          label: 'Lydian' },
  { value: 'mixolydian',      label: 'Mixolydian' },
  { value: 'aeolian',         label: 'Aeolian' },
  { value: 'locrian',         label: 'Locrian' },
  { value: 'harmonic_minor',  label: 'Harmonic Minor' },
  { value: 'melodic_minor',   label: 'Melodic Minor' },
  { value: 'natural_minor',   label: 'Natural Minor' },
  { value: 'ionian',          label: 'Ionian' },
  { value: 'pentatonic_major',label: 'Pentatonic Major' },
  { value: 'pentatonic_minor',label: 'Pentatonic Minor' },
  { value: 'blues',           label: 'Blues' },
  { value: 'chromatic',       label: 'Chromatic' },
];

const WAVEFORMS = {
  'Synth Pluck': 'sawtooth',
  'Piano':       'triangle',
  'Guitar':      'square',
  'Harp':        'sine',
};

const INSTRUMENTS = [
  { name: 'Synth Pluck', icon: '🎵' },
  { name: 'Piano',       icon: '🎹' },
  { name: 'Guitar',      icon: '🎸' },
  { name: 'Harp',        icon: '🎼' },
];

const KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const Home = () => {
  const navigate = useNavigate();

  // Build params
  const [tempo,      setTempo]      = useState(120);
  const [selectedKey,setSelectedKey]= useState('C');
  const [scale,      setScale]      = useState('major');
  const [noteCount,  setNoteCount]  = useState(16);
  const [bars,       setBars]       = useState(1);
  const [mood,       setMood]       = useState('');
  const [octave,     setOctave]     = useState(4);
  const [instrument, setInstrument] = useState('Piano');

  // State
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState('');
  const [result,     setResult]     = useState(null);
  const [isPlaying,  setIsPlaying]  = useState(false);

  // Audio refs
  const audioCtxRef   = useRef(null);
  const oscillatorsRef= useRef([]);
  const rafRef        = useRef(null);
  const [activeNoteIdx, setActiveNoteIdx] = useState(-1);

  useEffect(() => {
    if (!localStorage.getItem('token')) navigate('/');
  }, [navigate]);

  // ---- API call -----------------------------------------------------------

  const handleGenerate = async () => {
    if (!mood.trim()) { setError('Please enter a mood description'); return; }
    setError('');
    setLoading(true);
    stopPlayback();

    try {
      const data = await generateArpeggio({
        key:        selectedKey,
        scale,
        tempo:      parseInt(tempo),
        note_count: parseInt(noteCount),
        bars:       parseInt(bars),
        mood:       mood.trim(),
        octave:     parseInt(octave),
      });
      setResult(data);
    } catch (err) {
      setError(err.message || 'Generation failed — is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  // ---- Audio playback -----------------------------------------------------

  const stopPlayback = () => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    oscillatorsRef.current.forEach(osc => { try { osc.stop(0); } catch {} });
    oscillatorsRef.current = [];
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setIsPlaying(false);
    setActiveNoteIdx(-1);
  };

  const handlePlay = () => {
    // Stop is a clean stop; Play always resets to position 0
    if (isPlaying) { stopPlayback(); return; }
    if (!result)   return;

    // Tear down any lingering state before starting fresh
    stopPlayback();

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;

    const waveform       = WAVEFORMS[instrument] || 'sine';
    const secondsPerBeat = 60 / result.tempo;
    const startTime      = ctx.currentTime + 0.05;   // absolute t=0 for this playback

    // Capture displayNotes snapshot for the tracker closure
    const trackerNotes = displayNotes;

    const oscs = result.notes.map(note => {
      const start    = startTime + note.position * secondsPerBeat;
      const dur      = Math.max(0.05, note.duration * secondsPerBeat);
      const freq     = midiToFreq(note.pitch);
      const peakGain = (note.velocity / 127) * 0.25;

      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type            = waveform;
      osc.frequency.value = freq;
      osc.connect(gain);
      gain.connect(ctx.destination);

      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(peakGain, start + 0.01);
      gain.gain.setValueAtTime(peakGain, start + dur - 0.03);
      gain.gain.linearRampToValueAtTime(0, start + dur);

      osc.start(start);
      osc.stop(start + dur);
      return osc;
    });

    oscillatorsRef.current = oscs;
    setIsPlaying(true);

    // --- Tracker: highlight active bar via requestAnimationFrame ---
    const tick = () => {
      if (!audioCtxRef.current) return;
      const elapsedBeats = (audioCtxRef.current.currentTime - startTime) / secondsPerBeat;
      let active = -1;
      for (let i = 0; i < trackerNotes.length; i++) {
        const n = trackerNotes[i];
        if (elapsedBeats >= n.position && elapsedBeats < n.position + n.duration) {
          active = i;
          break;
        }
      }
      setActiveNoteIdx(active);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    // Auto-stop when sequence ends
    setTimeout(stopPlayback, (result.duration_seconds + 0.5) * 1000);
  };

  // ---- MIDI download ------------------------------------------------------

  const handleDownload = () => {
    if (!result) return;
    const raw   = atob(result.midi_base64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const blob  = new Blob([bytes], { type: 'audio/midi' });
    const url   = URL.createObjectURL(blob);
    const a     = document.createElement('a');
    a.href      = url;
    a.download  = `arpeggio_${result.key}_${result.scale}_${result.tempo}bpm.mid`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // ---- Logout -------------------------------------------------------------

  const handleLogout = () => {
    stopPlayback();
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/');
  };

  // ---- Visualization helpers ----------------------------------------------

  const vizNotes = result?.notes ?? [];
  const pitches  = vizNotes.map(n => n.pitch);
  const minPitch = pitches.length ? Math.min(...pitches) : 0;
  const maxPitch = pitches.length ? Math.max(...pitches) : 127;
  const pitchRange  = Math.max(1, maxPitch - minPitch);
  const avgVelocity = vizNotes.length
    ? Math.round(vizNotes.reduce((s, n) => s + n.velocity, 0) / vizNotes.length)
    : 0;
  const noteRangeLabel = vizNotes.length
    ? `${pitchToName(minPitch)} – ${pitchToName(maxPitch)}`
    : '—';

  // Downsample if more than 32 notes for readability
  const displayNotes = vizNotes.length > 32
    ? vizNotes.filter((_, i) => i % Math.ceil(vizNotes.length / 32) === 0).slice(0, 32)
    : vizNotes;

  // ---- Render -------------------------------------------------------------

  return (
    <div className="home-page">
      {/* Header */}
      <header className="home-header">
        <div className="home-logo">
          <Music size={24} strokeWidth={3} />
          <span>arpeggiator.ai</span>
        </div>
        <div className="home-nav">
          <button className="nav-btn" onClick={handleLogout}>
            <LogOut size={18} /> Log out
          </button>
        </div>
      </header>

      <main className="home-content">
        {/* ---- Left panel: Build ---- */}
        <div className="panel build-panel">
          <div className="panel-header">
            <span className="panel-icon">✨</span>
            <h2>Build Your Arpeggio</h2>
          </div>

          {/* Tempo */}
          <div className="control-group">
            <label>Tempo (BPM)</label>
            <div className="slider-container">
              <div className="slider-track">
                <div className="slider-fill" style={{ width: `${((tempo - 20) / (400 - 20)) * 100}%` }} />
                <input type="range" min="20" max="400" value={tempo}
                  onChange={e => setTempo(e.target.value)} className="slider" />
              </div>
              <span className="slider-value">{tempo}</span>
            </div>
          </div>

          {/* Key */}
          <div className="control-group">
            <label>Key</label>
            <div className="key-grid">
              {KEYS.map(k => (
                <button key={k}
                  className={`key-btn ${selectedKey === k ? 'selected' : ''}`}
                  onClick={() => setSelectedKey(k)}>
                  {k}
                </button>
              ))}
            </div>
          </div>

          {/* Scale */}
          <div className="control-group">
            <label>Scale</label>
            <div className="select-wrapper">
              <select value={scale} onChange={e => setScale(e.target.value)} className="scale-select">
                {SCALES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
              <ChevronDown className="select-icon" size={20} />
            </div>
          </div>

          {/* Notes + Octave */}
          <div className="two-col">
            <div className="control-group">
              <label>Notes in Pattern</label>
              <div className="slider-container">
                <div className="slider-track">
                  <div className="slider-fill" style={{ width: `${((noteCount - 4) / (32 - 4)) * 100}%` }} />
                  <input type="range" min="4" max="32" step="1" value={noteCount}
                    onChange={e => setNoteCount(parseInt(e.target.value))} className="slider" />
                </div>
                <span className="slider-value">{noteCount}</span>
              </div>
            </div>
            <div className="control-group">
              <label>Octave</label>
              <div className="slider-container">
                <div className="slider-track">
                  <div className="slider-fill" style={{ width: `${(octave / 8) * 100}%` }} />
                  <input type="range" min="0" max="8" value={octave}
                    onChange={e => setOctave(e.target.value)} className="slider" />
                </div>
                <span className="slider-value">{octave}</span>
              </div>
            </div>
          </div>

          {/* Bars (repetitions) */}
          <div className="control-group">
            <label>Bars <span style={{ fontWeight: 400, opacity: 0.6 }}>({bars} × {noteCount} = {bars * noteCount} notes total)</span></label>
            <div className="slider-container">
              <div className="slider-track">
                <div className="slider-fill" style={{ width: `${((bars - 1) / (8 - 1)) * 100}%` }} />
                <input type="range" min="1" max="8" step="1" value={bars}
                  onChange={e => setBars(parseInt(e.target.value))} className="slider" />
              </div>
              <span className="slider-value">{bars}</span>
            </div>
          </div>

          {/* Mood */}
          <div className="control-group">
            <label>Mood Description</label>
            <textarea
              value={mood}
              onChange={e => setMood(e.target.value)}
              placeholder="e.g., happy and energetic, calm and peaceful, dark and mysterious..."
              className="mood-textarea"
              rows={3}
            />
          </div>

          {error && <div className="error-banner">{error}</div>}

          <button className="generate-btn" onClick={handleGenerate} disabled={loading}>
            {loading
              ? <><span className="spinner" /> Generating…</>
              : <><Sparkles size={20} /> Generate Arpeggio</>}
          </button>
        </div>

        {/* ---- Right panel: Preview ---- */}
        <div className="panel preview-panel">
          <div className="panel-header">
            <span className="panel-icon">▶</span>
            <h2>Preview &amp; Export</h2>
          </div>

          <div className="visualization-section">
            <label>Pattern Sequence Visualization</label>

            {result ? (
              <>
                {/* Pitch bar chart with tracker */}
                <div className={`bar-chart${isPlaying ? ' playing' : ''}`}>
                  {displayNotes.map((note, i) => {
                    const isActive = activeNoteIdx === i;
                    return (
                      <div key={i} className={`bar-container${isActive ? ' bar-container-active' : ''}`}>
                        <div
                          className={`bar${isActive ? ' bar-active' : ''}`}
                          style={{
                            height: `${Math.max(8, ((note.pitch - minPitch) / pitchRange) * 85 + 8)}%`,
                            opacity: isPlaying && !isActive ? 0.25 : 0.45 + (note.velocity / 127) * 0.55,
                          }}
                        />
                        <span className={`bar-label${isActive ? ' bar-label-active' : ''}`}>
                          {pitchToName(note.pitch)}
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Stats */}
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Notes</div>
                    <div className="stat-value primary">{result.note_count}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Note Range</div>
                    <div className="stat-value">{noteRangeLabel}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Avg Velocity</div>
                    <div className="stat-value">{avgVelocity}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Duration</div>
                    <div className="stat-value">{result.duration_seconds.toFixed(1)} s</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">☀</div>
                <p>Generate a pattern to see the preview</p>
              </div>
            )}
          </div>

          {/* Instrument */}
          <div className="control-group">
            <label>Instrument Mode</label>
            <div className="instrument-grid">
              {INSTRUMENTS.map(inst => (
                <button
                  key={inst.name}
                  className={`instrument-btn ${instrument === inst.name ? 'selected' : ''}`}
                  onClick={() => setInstrument(inst.name)}>
                  <span className="instrument-icon">{inst.icon}</span>
                  {inst.name}
                </button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="action-buttons">
            <button className="preview-btn" onClick={handlePlay} disabled={!result}>
              {isPlaying
                ? <><Square size={18} /> Stop</>
                : <><Play  size={18} /> Play</>}
            </button>
            <button className="export-btn" onClick={handleDownload} disabled={!result}>
              <Download size={18} /> Export MIDI
            </button>
          </div>
        </div>
      </main>

      <footer className="home-footer">
        <div className="footer-text">© 2025 Arpeggiator.ai — Powered by AI</div>
        <div className="footer-links">
          <button className="footer-btn">☕</button>
          <button className="footer-btn">in</button>
          <button className="footer-btn">🎵</button>
          <button className="footer-btn">✉</button>
        </div>
      </footer>
    </div>
  );
};

export default Home;
