import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import imgLogo from '../assets/logo.png';
import imgIconMidi     from '../assets/7c89271e0d8c14fe49699b9d922aca42499e4e53.svg';
import imgIconSettings from '../assets/05432df0670198b2518870303f08acbab376865a.svg';
import imgIconLogout   from '../assets/b6bb2145169c29c365889f4c89ac97038f83e06c.svg';
import imgIconDownload from '../assets/7e755ab45c5f72177b624281f9d4d7b4fae6b60a.svg';
import { getFavorites, deleteFavorite, downloadFavorite } from '../services/api';

export default function SavedMidis() {
  const navigate = useNavigate();
  const dropdownRef = useRef(null);

  const [userName, setUserName]       = useState('');
  const [firstName, setFirstName]     = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [midis, setMidis]             = useState([]);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState('');
  const [deletingId, setDeletingId]       = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [downloadingId, setDownloadingId] = useState(null);

  // Auth guard
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) { navigate('/login'); return; }
    try {
      const user = JSON.parse(localStorage.getItem('user') || '{}');
      const name = user.full_name || user.name || user.email?.split('@')[0] || 'User';
      setUserName(name);
      setFirstName(name.split(' ')[0]);
    } catch { setUserName('User'); setFirstName('User'); }
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

  // Load saved MIDIs
  useEffect(() => {
    (async () => {
      try {
        const data = await getFavorites();
        setMidis(data);
      } catch (err) {
        setError(err.message || 'Failed to load saved MIDIs.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/');
  };

  const handleDelete = async () => {
    const id = confirmDeleteId;
    setConfirmDeleteId(null);
    setDeletingId(id);
    try {
      await deleteFavorite(id);
      setMidis(prev => prev.filter(m => m.id !== id));
    } catch {
      setError('Failed to delete. Please try again.');
    } finally {
      setDeletingId(null);
    }
  };

  const handleDownload = async (midi) => {
    setDownloadingId(midi.id);
    try {
      await downloadFavorite(midi.id, midi.filename);
    } catch {
      setError('Download failed. Please try again.');
    } finally {
      setDownloadingId(null);
    }
  };

  const formatDate = (iso) => {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const formatDuration = (s) => {
    if (s < 60) return `${Math.round(s)}s`;
    return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  };

  return (
    <div className="sm-root">
      <style>{`
        @import url('https://api.fontshare.com/v2/css?f[]=clash-grotesk@200,300,400,500,600,700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        .sm-root {
          font-family: 'ClashGrotesk-Variable', 'Clash Grotesk', sans-serif;
          min-height: 100vh;
          background: #0000ff;
          display: flex;
          flex-direction: column;
          padding: 11px 26px 40px;
          gap: 32px;
        }

        /* ── NAVBAR ── */
        .sm-navbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: 100%;
          position: relative;
          height: 64px;
          flex-shrink: 0;
          z-index: 200;
        }
        .sm-logo {
          display: flex;
          align-items: center;
          cursor: pointer;
          text-decoration: none;
        }
        .sm-logo img { height: 36px; width: auto; object-fit: contain; display: block; }

        .sm-nav-links {
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
        .sm-nav-link { font-size: 15px; font-weight: 500; color: #ffffff; text-decoration: none; transition: opacity 0.15s; }
        .sm-nav-link:hover { opacity: 0.75; }

        /* ── PROFILE DROPDOWN ── */
        .sm-user-wrap { position: relative; flex-shrink: 0; }
        .sm-user-btn {
          display: flex; align-items: center; justify-content: center;
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 50%;
          width: 44px; height: 44px;
          cursor: pointer; color: #ffffff;
          transition: background 0.18s, border-color 0.18s, transform 0.18s;
        }
        .sm-user-btn:hover {
          background: rgba(255,255,255,0.22);
          border-color: rgba(255,255,255,0.5);
          transform: scale(1.08);
        }
        .sm-dropdown {
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
          animation: smDropIn 0.18s cubic-bezier(0.22,1,0.36,1) both;
          padding: 5px 0 10px;
        }
        @keyframes smDropIn {
          from { opacity: 0; transform: translateY(-8px) scale(0.97); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        .sm-dropdown-header {
          display: flex; flex-direction: column; align-items: center;
          gap: 8px; padding: 20px 19px 16px;
          border-bottom: 1px solid rgba(255,255,255,0.15);
        }
        .sm-dropdown-avatar {
          width: 34px; height: 34px; border-radius: 50%;
          border: 1.5px solid rgba(255,255,255,0.4);
          display: flex; align-items: center; justify-content: center;
          color: #ffffff; background: rgba(255,255,255,0.1);
        }
        .sm-dropdown-hi { font-size: 14px; font-weight: 600; color: #ffffff; letter-spacing: -0.2px; }
        .sm-dropdown-item {
          display: flex; align-items: center; gap: 13px;
          padding: 12px 19px; font-size: 13px; font-weight: 500;
          font-family: 'Archivo', sans-serif; color: #ffffff;
          cursor: pointer; background: none; border: none;
          width: 100%; text-align: left; transition: background 0.12s;
        }
        .sm-dropdown-item:hover { background: rgba(255,255,255,0.1); }
        .sm-dropdown-item.logout { color: rgba(255,255,255,0.75); }
        .sm-dropdown-item.logout:hover { background: rgba(255,80,80,0.12); color: #ff8080; }
        .sm-dropdown-divider { height: 1px; background: rgba(255,255,255,0.12); margin: 5px 0; }
        .sm-dropdown-icon { width: 14px; height: 14px; flex-shrink: 0; opacity: 0.85; display: block; object-fit: contain; }

        /* ── PAGE HEADER ── */
        .sm-header {
          display: flex;
          flex-direction: column;
          gap: 8px;
          padding: 16px 0 8px;
          border-bottom: 1px solid rgba(255,255,255,0.12);
          margin-bottom: 4px;
        }
        .sm-heading {
          font-size: clamp(28px, 3.5vw, 46px);
          font-weight: 600;
          color: #ffffff;
          letter-spacing: -1.2px;
          line-height: 1.05;
        }
        .sm-subheading {
          font-size: 14px;
          font-weight: 400;
          color: rgba(255,255,255,0.5);
          font-family: 'Archivo', sans-serif;
          letter-spacing: 0.1px;
        }

        /* ── CONTENT ── */
        .sm-error {
          background: rgba(255,59,59,0.15);
          border: 1px solid rgba(255,59,59,0.4);
          border-radius: 10px;
          padding: 12px 16px;
          color: #ff8080;
          font-size: 13px;
          font-family: 'Archivo', sans-serif;
        }

        .sm-loading {
          color: rgba(255,255,255,0.5);
          font-family: 'Archivo', sans-serif;
          font-size: 15px;
          padding: 60px 0;
          text-align: center;
        }

        /* ── EMPTY STATE ── */
        .sm-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
          padding: 80px 20px;
          text-align: center;
        }
        .sm-empty-icon {
          width: 56px; height: 56px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          border: 1.5px solid rgba(255,255,255,0.2);
          display: flex; align-items: center; justify-content: center;
          color: rgba(255,255,255,0.4);
        }
        .sm-empty-title {
          font-size: 22px; font-weight: 600; color: #ffffff; letter-spacing: -0.5px;
        }
        .sm-empty-body {
          font-size: 14px; color: rgba(255,255,255,0.55);
          font-family: 'Archivo', sans-serif; max-width: 320px; line-height: 1.6;
        }
        .sm-empty-cta {
          display: inline-flex; align-items: center; gap: 8px;
          background: #ffffff; color: #0000ff;
          border: none; border-radius: 10px;
          padding: 12px 24px; font-size: 14px; font-weight: 600;
          font-family: 'Archivo', sans-serif; cursor: pointer;
          transition: opacity 0.15s, transform 0.15s;
          text-decoration: none;
        }
        .sm-empty-cta:hover { opacity: 0.9; transform: scale(1.03); }

        /* ── GRID ── */
        .sm-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 20px;
        }

        /* ── MIDI CARD ── */
        .sm-card {
          background: #0000ff;
          border: 2px solid rgba(255,255,255,0.35);
          border-radius: 16px;
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          transition: border-color 0.18s, transform 0.18s;
        }
        .sm-card:hover {
          border-color: rgba(255,255,255,0.7);
          transform: translateY(-2px);
        }

        .sm-card-top {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
        }
        .sm-card-mood {
          font-size: 17px;
          font-weight: 600;
          color: #ffffff;
          letter-spacing: -0.4px;
          line-height: 1.2;
          word-break: break-word;
        }
        .sm-card-date {
          font-size: 12px;
          font-weight: 400;
          color: rgba(255,255,255,0.5);
          font-family: 'Archivo', sans-serif;
          white-space: nowrap;
          flex-shrink: 0;
        }

        .sm-card-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }
        .sm-chip {
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(255,255,255,0.18);
          border-radius: 20px;
          padding: 4px 11px;
          font-size: 12px;
          font-weight: 500;
          color: rgba(255,255,255,0.85);
          font-family: 'Archivo', sans-serif;
        }
        .sm-chip.accent {
          background: rgba(191,255,62,0.12);
          border-color: rgba(191,255,62,0.3);
          color: #bfff3e;
        }

        .sm-card-actions {
          display: flex;
          gap: 10px;
          margin-top: auto;
        }
        .sm-btn-download {
          flex: 1;
          display: flex; align-items: center; justify-content: center; gap: 8px;
          height: 40px;
          background: #ffffff; color: #0000ff;
          border: none; border-radius: 8px;
          font-size: 13px; font-weight: 600;
          font-family: 'Archivo', sans-serif;
          cursor: pointer;
          transition: opacity 0.15s, transform 0.15s;
        }
        .sm-btn-download:hover { opacity: 0.9; transform: scale(1.02); }
        .sm-btn-download:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .sm-btn-delete {
          width: 40px; height: 40px; flex-shrink: 0;
          display: flex; align-items: center; justify-content: center;
          background: transparent;
          border: 1.5px solid rgba(255,255,255,0.25);
          border-radius: 8px;
          color: rgba(255,255,255,0.6);
          cursor: pointer;
          transition: background 0.15s, border-color 0.15s, color 0.15s;
        }
        .sm-btn-delete:hover {
          background: rgba(255,59,59,0.12);
          border-color: rgba(255,80,80,0.5);
          color: #ff8080;
        }
        .sm-btn-delete:disabled { opacity: 0.4; cursor: not-allowed; }

        .sm-btn-icon { width: 14px; height: 14px; flex-shrink: 0; }

        /* ── DELETE MODAL ── */
        .sm-modal-overlay {
          position: fixed; inset: 0;
          background: rgba(0,0,15,0.6);
          backdrop-filter: blur(6px);
          -webkit-backdrop-filter: blur(6px);
          display: flex; align-items: center; justify-content: center;
          z-index: 1000; padding: 20px;
          animation: smFadeIn 0.18s ease both;
        }
        @keyframes smFadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        .sm-modal {
          background: #ffffff;
          border-radius: 20px;
          padding: 28px 28px 24px;
          width: 100%; max-width: 340px;
          display: flex; flex-direction: column; gap: 6px;
          box-shadow: 0 24px 60px rgba(0,0,0,0.35);
          animation: smSlideUp 0.22s cubic-bezier(0.22,1,0.36,1) both;
        }
        @keyframes smSlideUp {
          from { opacity: 0; transform: translateY(16px) scale(0.96); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        .sm-modal-title {
          font-size: 22px; font-weight: 600; color: #000000;
          letter-spacing: -0.5px; margin-bottom: 4px;
        }
        .sm-modal-body {
          font-size: 15px; color: #444455;
          font-family: 'Archivo', sans-serif; line-height: 1.55;
          margin-bottom: 18px;
        }
        .sm-modal-actions { display: flex; gap: 8px; }
        .sm-modal-cancel {
          flex: 1; height: 42px;
          background: #f0f0f5; border: none; border-radius: 10px;
          color: #060511; font-size: 14px; font-weight: 600;
          font-family: 'Archivo', sans-serif;
          cursor: pointer; transition: background 0.15s;
        }
        .sm-modal-cancel:hover { background: #e4e4ef; }
        .sm-modal-confirm {
          flex: 1; height: 42px;
          background: #ff3b3b; border: none; border-radius: 10px;
          color: #ffffff; font-size: 14px; font-weight: 600;
          font-family: 'Archivo', sans-serif;
          cursor: pointer; transition: background 0.15s;
        }
        .sm-modal-confirm:hover { background: #e62e2e; }

        /* ── FOOTER ── */
        .sm-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 12px;
          border-top: 1px solid rgba(255,255,255,0.1);
          margin-top: auto;
        }
        .sm-footer-copy { font-size: 12px; color: rgba(255,255,255,0.4); font-family: 'Archivo', sans-serif; }
        .sm-footer-coffee { font-size: 12px; color: rgba(255,255,255,0.6); font-family: 'Archivo', sans-serif; text-decoration: none; }
        .sm-footer-coffee:hover { color: #ffffff; }

        @media (max-width: 640px) {
          .sm-root { padding: 11px 16px 32px; }
          .sm-nav-links { display: none; }
          .sm-grid { grid-template-columns: 1fr; }
        }
      `}</style>

      {/* ── NAVBAR ── */}
      <nav className="sm-navbar">
        <Link to="/" className="sm-logo" aria-label="Arpeggiate home">
          <img src={imgLogo} alt="Arpeggiate" />
        </Link>

        <div className="sm-nav-links">
          <Link to="/build" className="sm-nav-link">Home</Link>
          <a href="/#create" className="sm-nav-link">Explore</a>
          <a href="/#create" className="sm-nav-link">Create</a>
        </div>

        <div className="sm-user-wrap" ref={dropdownRef}>
          <button
            className="sm-user-btn"
            onClick={() => setDropdownOpen(o => !o)}
            aria-expanded={dropdownOpen}
            aria-label="Profile menu"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
            </svg>
          </button>

          {dropdownOpen && (
            <div className="sm-dropdown">
              <div className="sm-dropdown-header">
                <div className="sm-dropdown-avatar">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                  </svg>
                </div>
                <span className="sm-dropdown-hi">Hi, {firstName}</span>
              </div>
              <button className="sm-dropdown-item">
                <svg className="sm-dropdown-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                </svg>
                View Profile
              </button>
              <button className="sm-dropdown-item" onClick={() => navigate('/saved')}>
                <img src={imgIconMidi} alt="" className="sm-dropdown-icon" />
                Saved MIDIs
              </button>
              <button className="sm-dropdown-item">
                <svg className="sm-dropdown-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="5" width="20" height="14" rx="2"/>
                  <line x1="2" y1="10" x2="22" y2="10"/>
                  <line x1="6" y1="15" x2="10" y2="15"/>
                </svg>
                Billing
              </button>
              <button className="sm-dropdown-item">
                <img src={imgIconSettings} alt="" className="sm-dropdown-icon" />
                Settings
              </button>
              <div className="sm-dropdown-divider" />
              <button className="sm-dropdown-item logout" onClick={handleLogout}>
                <img src={imgIconLogout} alt="" className="sm-dropdown-icon" />
                Log out
              </button>
            </div>
          )}
        </div>
      </nav>

      {/* ── PAGE HEADER ── */}
      <div className="sm-header">
        <h1 className="sm-heading">Saved MIDIs</h1>
        <p className="sm-subheading">
          {!loading && midis.length > 0
            ? `${midis.length} saved file${midis.length !== 1 ? 's' : ''}`
            : 'Your saved arpeggios'}
        </p>
      </div>

      {/* ── CONTENT ── */}
      {error && <div className="sm-error">{error}</div>}

      {loading ? (
        <div className="sm-loading">Loading your saved MIDIs…</div>
      ) : midis.length === 0 ? (
        <div className="sm-empty">
          <div className="sm-empty-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
            </svg>
          </div>
          <p className="sm-empty-title">No saved MIDIs yet</p>
          <p className="sm-empty-body">
            Generate an arpeggio and tap the heart icon to save it here for later.
          </p>
          <Link to="/build" className="sm-empty-cta">
            Start building
          </Link>
        </div>
      ) : (
        <div className="sm-grid">
          {midis.map(midi => (
            <div className="sm-card" key={midi.id}>
              <div className="sm-card-top">
                <span className="sm-card-mood">{midi.filename}</span>
                <span className="sm-card-date">{formatDate(midi.created_at)}</span>
              </div>

              <div className="sm-card-chips">
                <span className="sm-chip accent">{midi.key} {midi.scale.replace('_', ' ')}</span>
                <span className="sm-chip">{midi.tempo} BPM</span>
                <span className="sm-chip">{midi.note_count} notes</span>
                <span className="sm-chip">{formatDuration(midi.duration_seconds)}</span>
              </div>

              <div className="sm-card-actions">
                <button
                  className="sm-btn-download"
                  onClick={() => handleDownload(midi)}
                  disabled={downloadingId === midi.id}
                >
                  <img src={imgIconDownload} alt="" className="sm-btn-icon" />
                  {downloadingId === midi.id ? 'Downloading…' : 'Export MIDI'}
                </button>
                <button
                  className="sm-btn-delete"
                  onClick={() => setConfirmDeleteId(midi.id)}
                  disabled={deletingId === midi.id}
                  aria-label="Delete"
                  title="Delete"
                >
                  {deletingId === midi.id ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="9" strokeDasharray="56" strokeDashoffset="14"/>
                    </svg>
                  ) : (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <polyline points="3 6 5 6 21 6"/>
                      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                      <path d="M10 11v6M14 11v6"/>
                      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                    </svg>
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DELETE CONFIRM MODAL ── */}
      {confirmDeleteId !== null && (
        <div className="sm-modal-overlay" onClick={() => setConfirmDeleteId(null)}>
          <div className="sm-modal" onClick={e => e.stopPropagation()}>
            <p className="sm-modal-title">Delete this MIDI?</p>
            <p className="sm-modal-body">This file will be permanently removed from your saved MIDIs and cannot be recovered.</p>
            <div className="sm-modal-actions">
              <button className="sm-modal-cancel" onClick={() => setConfirmDeleteId(null)}>Cancel</button>
              <button className="sm-modal-confirm" onClick={handleDelete}>Delete</button>
            </div>
          </div>
        </div>
      )}

      {/* ── FOOTER ── */}
      <footer className="sm-footer">
        <span className="sm-footer-copy">© 2025 Arpeggiate.ai - Powered by AI</span>
        <a href="https://buymeacoffee.com" target="_blank" rel="noopener noreferrer" className="sm-footer-coffee">
          Buy me a coffee
        </a>
      </footer>
    </div>
  );
}
