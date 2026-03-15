import React, { useEffect, useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';

// Local assets
import imgMusicalNotes from './assets/c9389c586242c8cafab5ab9bc8561f822b8f25f3.png';
import imgHeroDevice from './assets/83850e6a8a3eea5671ee72380a886a6c309fc1af.png';
import imgHeroWaveYellow from './assets/69b277268d18f3cf7cf33c3a25ae1126882391e1.svg';
import imgHeroFrame from './assets/f2cdd9c29e9f965c22315619ffd263ee22a43e13.svg';
import imgPlayButton from './assets/ed95f6de6f5106eb85102e2dae4510d8836d6d94.svg';
import imgPlaybackIcon from './assets/0d0623da6401ac67a0343a6476c155dff632d6c7.svg';
import imgWaveformMask from './assets/7ff68f2d89b75475398b120699a93bb279e9b1c2.png';
import imgAudioLine from './assets/d2729f8eb89dc66facc6bbb2efd521680c1cedd6.svg';
import imgFounderPhoto from './assets/a83aa3225e8cb2ae8048839e692c745d6a1e5a8e.png';
import imgMusicianPhoto from './assets/23189a40c33d558e67738a5e2bde8f87af042bf9.png';
import imgUserAvatar from './assets/7fd7b2055bb2f556381513a55b6951492f6e47d0.png';
import imgLogomark from './assets/479f4df41573c65a044d4f54ba8995f9e6360493.svg';
import imgIconCoffee from './assets/bc44376adc86e04210464f6978ca16d7112d3b24.svg';
import imgIconGithub from './assets/4c445b99e3df7faabeb5c02da450b2d1abc92b84.svg';
import imgIconTwitter from './assets/b842c300410c810b2c9aba1587007728fc3188e8.svg';
import imgLogo from './assets/logo.png';
import imgContainerMask from './assets/62fc9d434160142e81cdad28bc186bf3a3496341.svg';
import imgContentMask from './assets/d87b360cd7dacddb6e9d077c668b09662b5938a4.svg';
import imgWsLogoFl from './assets/ws_logo_fl.png';
import imgWsLogoLogic from './assets/ws_logo_logic.png';
import imgWsLogoAbleton from './assets/ws_logo_ableton.png';
import imgWsLogoProtools from './assets/ws_logo_protools.png';
import imgWsLogo5 from './assets/ws_logo_5.png';
import imgWsLogo6 from './assets/ws_logo_6.png';

export default function ArpeggiatorLanding() {
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      navigate('/build');
    }
  }, [navigate]);

  // Parallax refs
  const notesRef    = useRef(null);
  const spheresRef  = useRef(null);
  const heroTextRef = useRef(null);
  const visualsRef  = useRef(null);

  // Workstations carousel refs
  const wsContainerRef = useRef(null);
  const wsTrackRef     = useRef(null);

  // Parallax scroll handler
  useEffect(() => {
    let raf;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const y = window.scrollY;
        if (notesRef.current)
          notesRef.current.style.transform = `translateY(${y * 0.18}px)`;
        if (spheresRef.current)
          spheresRef.current.style.transform = `translateX(-50%) translateY(${y * 0.07}px)`;
        if (heroTextRef.current)
          heroTextRef.current.style.transform = `translateY(${y * -0.06}px)`;
        if (visualsRef.current)
          visualsRef.current.style.transform = `translateY(${y * 0.04}px)`;
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll(); // apply correct transform immediately on mount
    return () => { window.removeEventListener('scroll', onScroll); cancelAnimationFrame(raf); };
  }, []);

  // Scroll-reveal via IntersectionObserver
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  // Cycling export format
  const exportFormats = ['.wav', '.mp4', '.mid'];
  const [formatIdx, setFormatIdx] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setFormatIdx(i => (i + 1) % exportFormats.length), 2000);
    return () => clearInterval(id);
  }, []);

  // Compatible Workstations — infinite loop carousel
  const workstations = [
    { name: 'FL Studio',  img: imgWsLogoFl       },
    { name: 'Logic Pro',  img: imgWsLogoLogic     },
    { name: 'Ableton',    img: imgWsLogoAbleton   },
    { name: 'Pro Tools',  img: imgWsLogoProtools  },
    { name: 'DAW 5',      img: imgWsLogo5         },
    { name: 'DAW 6',      img: imgWsLogo6         },
  ];
  const WS_N    = workstations.length;
  const wsItems = [...workstations, ...workstations, ...workstations]; // triple for seamless loop
  const WS_SLOT = 208; // 160px slot + 48px gap

  // Start in the middle copy so we can scroll in both directions with no edge
  const [activeWs, setActiveWs] = useState(WS_N);

  // Center logo at `idx` by reading actual DOM positions (handles any gap/size variance)
  const wsSetPos = (idx, animated) => {
    if (!wsContainerRef.current || !wsTrackRef.current) return;
    const el = wsTrackRef.current.children[idx];
    if (!el) return;
    const containerW = wsContainerRef.current.offsetWidth;
    const elCenter = el.offsetLeft + el.offsetWidth / 2;
    const offset = containerW / 2 - elCenter;
    if (!animated) wsTrackRef.current.style.transition = 'none';
    wsTrackRef.current.style.transform = `translateX(${offset}px)`;
  };

  // Set initial position without animation
  useEffect(() => {
    wsSetPos(WS_N, false);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (wsTrackRef.current) wsTrackRef.current.style.transition = '';
    }));
    // Recentre on resize
    const onResize = () => wsSetPos(WS_N, false);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Auto-advance
  useEffect(() => {
    const id = setInterval(() => setActiveWs(i => i + 1), 2500);
    return () => clearInterval(id);
  }, []);

  // Slide to active; reset silently when entering the third copy
  useEffect(() => {
    wsSetPos(activeWs, true);
    if (activeWs >= WS_N * 2) {
      const t = setTimeout(() => {
        const resetTo = activeWs - WS_N;
        if (wsTrackRef.current) wsTrackRef.current.style.transition = 'none';
        wsSetPos(resetTo, false);
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (wsTrackRef.current) wsTrackRef.current.style.transition = '';
        }));
        setActiveWs(resetTo);
      }, 700); // after transition finishes
      return () => clearTimeout(t);
    }
  }, [activeWs]);

  const genres = ['baile-funk', 'pop', 'deep-house', 'afro-house', 'amapiano', 'gqom', 'soca', 'bouyon', 'kompa'];
  const [selectedGenres, setSelectedGenres] = useState(['baile-funk', 'deep-house', 'amapiano', 'gqom', 'soca', 'bouyon']);

  const toggleGenre = (genre) => {
    setSelectedGenres(prev =>
      prev.includes(genre) ? prev.filter(g => g !== genre) : [...prev, genre]
    );
  };

  return (
    <div className="landing-root">
      <style>{`
        @import url('https://api.fontshare.com/v2/css?f[]=clash-grotesk@200,300,400,500,600,700&display=swap');

        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          overflow-x: hidden;
        }

        .landing-root {
          font-family: 'ClashGrotesk-Variable', 'Clash Grotesk', sans-serif;
          background: #0000ff;
          color: #ffffff;
          min-height: 100vh;
          overflow-x: hidden;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 46px;
          padding: 12px 20px 20px;
        }

        /* ─── NAVBAR ─── */
        .lp-navbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: 100%;
          position: relative;
          height: 64px;
          flex-shrink: 0;
        }

        /* Logo: 5-bar arch (short-medium-tall-medium-short), bottom-aligned */
        .lp-logo {
          display: flex;
          align-items: center;
          cursor: pointer;
          text-decoration: none;
        }

        .lp-logo img {
          height: 36px;
          width: auto;
          object-fit: contain;
          display: block;
        }

        .lp-nav-links {
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
        }

        .lp-nav-link {
          font-size: 15px;
          font-weight: 500;
          color: #ffffff;
          text-decoration: none;
          white-space: nowrap;
          transition: opacity 0.15s;
        }

        .lp-nav-link:hover {
          opacity: 0.75;
        }

        .lp-nav-signin {
          background: #d9ff00;
          color: #03061f;
          font-size: 15px;
          font-weight: 500;
          padding: 10px 22px;
          border-radius: 40px;
          text-decoration: none;
          white-space: nowrap;
          transition: opacity 0.15s;
          border: none;
          cursor: pointer;
          display: inline-block;
        }

        .lp-nav-signin:hover {
          opacity: 0.88;
        }

        /* ─── HERO ─── */
        .lp-hero {
          position: relative;
          width: 100%;
          min-height: calc(100vh - 100px);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 0;
          overflow: visible;
          padding-bottom: 120px;
        }

        /* Decorative background musical notes — behind everything */
        .lp-hero-notes {
          position: absolute;
          inset: 0;
          pointer-events: none;
          z-index: 0;
          isolation: isolate;
        }

        .lp-hero-notes img {
          position: absolute;
          width: 651px;
          height: 507px;
          opacity: 0.12;
          transform: rotate(-2.87deg);
          object-fit: cover;
        }

        .lp-hero-notes img:first-child {
          left: 15.5vw;
          top: -40px;
        }

        .lp-hero-notes img:last-child {
          left: 38.3vw;
          top: 131px;
        }

        /* Spheres / blobs behind headline */
        .lp-hero-spheres {
          position: absolute;
          left: 50%;
          top: 27px;
          transform: translateX(-50%);
          width: 611px;
          height: 590px;
          max-width: 95vw;
          object-fit: contain;
          pointer-events: none;
          z-index: 1;
        }

        /* Hero text stack */
        .lp-hero-text {
          position: relative;
          z-index: 2;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 10px;
          max-width: 820px;
          width: 100%;
          margin-top: 40px;
        }

        .lp-hero-headline {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0;
        }

        .lp-hero-unleash-row {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: -10px;
        }

        .lp-hero-dash {
          background: #d9d9d9;
          height: 8px;
          width: 22px;
          border-radius: 100px;
          flex-shrink: 0;
          opacity: 0.7;
        }

        .lp-hero-unleash {
          font-size: clamp(46px, 7.5vw, 105px);
          font-weight: 900;
          line-height: 0.85;
          letter-spacing: 6px;
          color: #d9ff00;
          letter-spacing: -3px;
          text-align: center;
          white-space: nowrap;
        }

        .lp-hero-potential {
          font-size: clamp(24px, 3.8vw, 56px);
          font-weight: 700;
          line-height: 0.8;
          color: #ffffff;
          text-align: center;
          white-space: nowrap;
          letter-spacing: -1px;
        }

        .lp-hero-sub {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 20px;
        }

        .lp-hero-subtitle {
          font-size: clamp(15px, 1.5vw, 21px);
          font-weight: 400;
          text-align: center;
          color: rgba(255,255,255,0.85);
          line-height: 1.5;
          max-width: 480px;
        }

        .lp-hero-subtitle span {
          color: #bfff3e;
          font-weight: 500;
        }

        .lp-hero-cta {
          background: #bfff3e;
          color: #060511;
          font-size: clamp(15px, 1.3vw, 20px);
          font-weight: 500;
          padding: 14px 36px;
          border-radius: 40px;
          text-decoration: none;
          white-space: nowrap;
          transition: background 0.25s, color 0.25s, transform 0.25s, box-shadow 0.25s;
          display: inline-block;
          letter-spacing: -0.2px;
        }

        .lp-hero-cta:hover {
          background: #a8e800;
          transform: translateY(-3px);
        }

        /* Hero product image + waves — all inside this container */
        .lp-hero-visuals {
          position: relative;
          width: 100%;
          height: 420px;
          margin-top: 20px;
          z-index: 2;
          overflow: visible;
        }

        /* Combined wave (both left + right shapes merged into one SVG) */
        .lp-hero-wave-yellow {
          position: absolute;
          left: -30px;
          top: -10px;
          width: 100vw;
          height: 100%;
          pointer-events: none;
          z-index: 1;
          display: block;
          border: none;
          outline: none;
        }

        .lp-hero-device {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          top: 0;
          width: 700px;
          height: 410px;
          object-fit: contain;
          pointer-events: none;
          z-index: 3;
        }

        /* Export label — right of device */
        .lp-export-badges {
          position: absolute;
          left: calc(50% + 230px);
          top: 240px;
          z-index: 4;
          pointer-events: none;
          white-space: nowrap;
        }

        .lp-export-badge {
          font-size: 46px;
          font-weight: 500;
          color: #ffffff;
          display: flex;
          align-items: baseline;
          gap: 8px;
          line-height: 1;
          margin: 0;
          padding: 0;
        }

        @keyframes formatSlideUp {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .lp-export-format {
          display: inline-block;
          color: #d9ff00;
          font-style: italic;
          font-weight: 900;
          text-decoration: underline;
          text-decoration-thickness: 1px;
          text-underline-offset: 4px;
          animation: formatSlideUp 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        /* ─── CREATE SECTION ─── */
        .lp-create-section {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 31px;
          width: 100%;
          padding: 0 16px;
        }

        .lp-create-heading {
          font-size: clamp(16px, 1.9vw, 30px);
          font-weight: 500;
          text-align: center;
          color: #eaeaea;
        }

        .lp-create-heading em {
          font-style: normal;
          font-weight: 700;
          color: #bfff3e;
        }

        .lp-create-body {
          display: flex;
          gap: 46px;
          align-items: center;
          width: 100%;
        }

        /* Genre card stack (visual placeholder) */
        .lp-genre-visual {
          position: relative;
          width: 520px;
          height: 370px;
          flex-shrink: 0;
        }

        .lp-genre-visual-bg1 {
          position: absolute;
          left: 0;
          top: 0;
          width: 500px;
          height: 356px;
          background: #bfff3e;
          border-radius: 31px;
          box-shadow: 0 4px 4px rgba(0,0,0,0.25);
        }

        .lp-genre-visual-bg2 {
          position: absolute;
          left: 17px;
          top: 13px;
          width: 500px;
          height: 356px;
          background: #ffffff;
          border-radius: 31px;
          box-shadow: 0 4px 4px rgba(0,0,0,0.25);
        }

        /* Genre pills + player */
        .lp-create-right {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 36px;
          height: 100%;
        }

        .lp-genres-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          justify-content: center;
          max-width: 515px;
          width: 100%;
        }

        .lp-genre-pill {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 11px 22px;
          border-radius: 50px;
          font-size: clamp(12px, 1vw, 16px);
          font-weight: 500;
          text-align: center;
          cursor: pointer;
          border: 1.5px solid #ffffff;
          background: transparent;
          color: #ffffff;
          font-family: inherit;
          transition: background 0.15s, color 0.15s, border-color 0.15s;
          user-select: none;
        }

        .lp-genre-pill:hover {
          background: rgba(255,255,255,0.12);
        }

        .lp-genre-pill.selected {
          background: #ffffff;
          color: #0000ff;
          border-color: #ffffff;
        }

        .lp-genre-pill.selected:hover {
          background: #e0e0ff;
        }

        .lp-audio-player {
          display: flex;
          align-items: center;
          gap: 31px;
          width: 100%;
          background: linear-gradient(to right, rgba(188,255,53,0.51), rgba(58,58,201,0.59));
          border: 1px solid #ffffff;
          border-radius: 100px;
          padding: 8px 31px;
        }

        .lp-audio-info {
          display: flex;
          flex: 1;
          flex-direction: column;
          gap: 5px;
          align-items: center;
        }

        .lp-audio-title {
          font-size: 16px;
          font-weight: 500;
          color: #ffffff;
          white-space: pre;
        }

        .lp-audio-waveform {
          width: 100%;
          height: 47px;
          position: relative;
          overflow: hidden;
        }

        .lp-audio-line {
          width: 100%;
          height: 2px;
          position: relative;
        }

        .lp-audio-line img {
          width: 100%;
          height: 100%;
          object-fit: fill;
        }

        .lp-play-btn {
          width: 50px;
          height: 50px;
          flex-shrink: 0;
          cursor: pointer;
          transition: opacity 0.15s;
        }

        .lp-play-btn:hover {
          opacity: 0.85;
        }

        .lp-play-btn img {
          width: 100%;
          height: 100%;
          object-fit: contain;
        }

        /* ─── VIDEO / DEMO SECTION ─── */
        .lp-demo-section {
          background: #0000ba;
          border-radius: 31px;
          width: 100%;
          height: 300px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          cursor: pointer;
          transition: background 0.2s;
          overflow: hidden;
          position: relative;
        }

        .lp-demo-section:hover {
          background: #0000cc;
        }

        .lp-demo-play-icon {
          width: 120px;
          height: 120px;
          object-fit: contain;
          filter: drop-shadow(0 0 31px rgba(191,255,62,0.4));
        }

        /* ─── WHY CHOOSE SECTION ─── */
        .lp-why-section {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 22px;
          width: 100%;
          padding: 24px 0;
        }

        .lp-why-heading {
          font-size: clamp(16px, 1.9vw, 28px);
          font-weight: 700;
          color: #ffffff;
          text-align: center;
          letter-spacing: -0.37px;
          width: 100%;
        }

        .lp-why-heading span {
          font-weight: 500;
        }

        .lp-bento-grid {
          display: flex;
          flex-direction: column;
          gap: 19px;
          width: 100%;
        }

        .lp-bento-row {
          display: flex;
          gap: 19px;
          align-items: stretch;
          width: 100%;
          min-height: 280px;
        }

        .lp-bento-card {
          background: #0000ff;
          border: 2px solid #ffffff;
          border-radius: 12px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          padding-top: 25px;
          position: relative;
          min-height: 280px;
        }

        .lp-bento-card.flex-1 {
          flex: 3;
        }

        .lp-bento-card.w480 {
          flex: 2;
        }

        .lp-bento-card.w480sq {
          flex: 2;
        }

        .lp-bento-card.w642 {
          flex: 3;
        }

        .lp-bento-text {
          display: flex;
          flex-direction: column;
          gap: 6px;
          padding: 0 25px;
          width: 100%;
          color: #ffffff;
          flex-shrink: 0;
        }

        .lp-bento-title {
          font-size: 17px;
          font-weight: 900;
          line-height: 1.2;
          letter-spacing: -0.22px;
          color: #ffffff;
        }

        .lp-bento-desc {
          font-size: 12px;
          font-weight: 500;
          line-height: 1.4;
          color: #ffffff;
        }

        /* Bento card image areas */
        .lp-bento-img-area {
          flex: 1;
          width: 100%;
          position: relative;
          overflow: hidden;
          min-height: 234px;
        }

        /* Emotion analysis UI snippet */
        .lp-bento-ui-emotion {
          position: absolute;
          bottom: -24px;
          right: -20px;
          width: 260px;
          height: 230px;
          background: #0c0cda;
          border: 2px solid rgba(255,255,255,0.16);
          border-radius: 19px;
          overflow: hidden;
          box-shadow: 2px 21px 22px 0px rgba(0,0,0,0.06);
        }

        .lp-emotion-user-row {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 23px;
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
        }

        .lp-emotion-avatar {
          width: 44px;
          height: 44px;
          border-radius: 50%;
          object-fit: cover;
          flex-shrink: 0;
        }

        .lp-emotion-label {
          font-size: 20px;
          font-weight: 900;
          color: #ffffff;
          letter-spacing: -0.26px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .lp-emotion-bars {
          position: absolute;
          left: 23px;
          right: 23px;
          top: 86px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .lp-emotion-bar {
          height: 31px;
          border-radius: 6px;
          background: rgba(255,255,255,0.2);
          opacity: 0.5;
        }

        .lp-emotion-bar.active {
          background: rgba(170,255,0,0.6);
          opacity: 1;
        }

        /* Command keys UI snippet */
        .lp-bento-ui-keys {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -30%);
          display: flex;
          gap: 16px;
        }

        .lp-key-box {
          width: 109px;
          height: 109px;
          background: #0c0cda;
          border: 1px solid rgba(255,255,255,0.16);
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 70px;
          color: #ffffff;
          font-weight: 400;
          box-shadow: 1px 4px 9px 0px rgba(0,0,0,0.08);
          overflow: hidden;
        }

        /* Collaboration snippet */
        .lp-bento-ui-collab {
          position: absolute;
          bottom: -24px;
          left: -20px;
          width: 260px;
          height: 230px;
          background: #0d0dda;
          border: 2px solid rgba(255,255,255,0.16);
          border-radius: 19px;
          overflow: hidden;
          box-shadow: 2px 21px 22px 0px rgba(0,0,0,0.06);
        }

        .lp-collab-buttons {
          display: flex;
          align-items: center;
          gap: 12px;
          position: absolute;
          top: 22px;
          right: 22px;
        }

        .lp-collab-btn-ghost {
          height: 25px;
          border-radius: 6px;
          background: rgba(255,255,255,0.2);
          opacity: 0.5;
        }

        .lp-collab-btn-solid {
          background: #679940;
          height: 37px;
          border-radius: 6px;
          padding: 6px 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 16px;
          font-weight: 500;
          color: #000000;
          white-space: nowrap;
        }

        .lp-collab-content-bg {
          position: absolute;
          background: rgba(255,255,255,0.2);
          inset: 86px 22px 25px -2px;
          border-radius: 6px;
          opacity: 0.5;
        }

        /* ─── FOUNDER SECTION ─── */
        .lp-founder-section {
          display: flex;
          gap: 62px;
          align-items: center;
          justify-content: center;
          width: 100%;
        }

        .lp-founder-card {
          display: flex;
          gap: 24px;
          align-items: center;
          padding: 32px;
          border: 2px solid rgba(255,255,255,0.9);
          border-radius: 37px;
          box-shadow: 2px 8px 18px 0px rgba(0,0,0,0.08);
          flex-shrink: 0;
          overflow: hidden;
          background: transparent;
          position: relative;
        }

        .lp-founder-photo {
          width: 120px;
          height: 152px;
          border-radius: 27px;
          object-fit: cover;
          flex-shrink: 0;
        }

        .lp-founder-info {
          display: flex;
          gap: 52px;
          align-items: center;
          flex-shrink: 0;
        }

        .lp-founder-text {
          display: flex;
          flex-direction: column;
          gap: 9px;
          color: #ffffff;
          white-space: nowrap;
        }

        .lp-founder-role {
          font-size: 28px;
          font-weight: 700;
          line-height: 1.1;
          letter-spacing: -0.43px;
        }

        .lp-founder-name {
          font-size: 25px;
          font-weight: 500;
          letter-spacing: -0.18px;
        }

        .lp-founder-title {
          font-size: 15px;
          font-weight: 500;
          letter-spacing: -0.1px;
        }

        .lp-founder-socials {
          display: flex;
          flex-direction: column;
          gap: 31px;
          align-items: flex-start;
        }

        .lp-social-icon-box {
          background: rgba(29,41,61,0.4);
          border-radius: 12px;
          width: 50px;
          height: 50px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-decoration: none;
          transition: opacity 0.15s;
        }

        .lp-social-icon-box:hover {
          opacity: 0.75;
        }

        .lp-social-icon-box img {
          width: 25px;
          height: 25px;
          object-fit: contain;
        }

        .lp-founder-byline {
          display: flex;
          flex-direction: column;
          gap: 12px;
          max-width: 431px;
          flex-shrink: 0;
        }

        .lp-founder-headline {
          font-size: clamp(18px, 2.4vw, 38px);
          font-weight: 700;
          line-height: 1.05;
          letter-spacing: -0.48px;
          color: #ffffff;
        }

        .lp-founder-desc {
          font-size: clamp(11px, 1.1vw, 18px);
          font-weight: 500;
          line-height: 1.4;
          letter-spacing: -0.11px;
          color: #ffffff;
        }

        /* ─── TESTIMONIAL SECTION ─── */
        .lp-testimonial-section {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 26px;
          width: 100%;
          padding: 40px 0;
          overflow: hidden;
        }

        .lp-testimonial-quote {
          font-size: clamp(14px, 1.6vw, 26px);
          font-weight: 400;
          line-height: 1.1;
          text-align: center;
          color: #ffffff;
          letter-spacing: -1px;
          max-width: 984px;
        }

        .lp-testimonial-author {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .lp-testimonial-avatar {
          width: 50px;
          height: 50px;
          border-radius: 999px;
          object-fit: cover;
          flex-shrink: 0;
        }

        .lp-testimonial-author-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .lp-testimonial-brand-row {
          display: flex;
          align-items: center;
          gap: 6px;
          height: 25px;
        }

        .lp-testimonial-brand-logo {
          width: 25px;
          height: 25px;
          object-fit: contain;
        }

        .lp-testimonial-brand-name {
          font-size: 22px;
          font-weight: 400;
          color: #ffffff;
          white-space: nowrap;
          letter-spacing: 0;
        }

        .lp-testimonial-role {
          font-size: 13px;
          font-weight: 500;
          color: rgba(255,255,255,0.6);
          white-space: nowrap;
        }

        /* ─── FOOTER ─── */
        .lp-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: 100%;
          padding: 25px 32px;
          background: #0000ef;
          border-radius: 16px;
          min-height: 90px;
        }

        .lp-footer-copy {
          font-size: 14px;
          font-weight: 400;
          color: #ffffff;
          letter-spacing: -0.15px;
          white-space: nowrap;
        }

        .lp-footer-links {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .lp-footer-link-pill {
          display: flex;
          align-items: center;
          gap: 8px;
          background: #0c0cda;
          border-radius: 10px;
          padding: 8px 16px;
          height: 36px;
          text-decoration: none;
          transition: opacity 0.15s;
        }

        .lp-footer-link-pill:hover {
          opacity: 0.8;
        }

        .lp-footer-link-pill img {
          width: 16px;
          height: 16px;
          object-fit: contain;
        }

        .lp-footer-link-pill span {
          font-size: 14px;
          color: #cad5e2;
          white-space: nowrap;
        }

        .lp-footer-icon-link {
          background: #0c0cda;
          border-radius: 10px;
          width: 40px;
          height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-decoration: none;
          transition: opacity 0.15s;
        }

        .lp-footer-icon-link:hover {
          opacity: 0.8;
        }

        .lp-footer-icon-link img {
          width: 20px;
          height: 20px;
          object-fit: contain;
        }

        /* ─── PARALLAX: prep GPU layers ─── */
        .lp-hero-notes,
        .lp-hero-spheres,
        .lp-hero-text,
        .lp-hero-visuals {
          will-change: transform;
        }

        /* ─── KEYFRAMES ─── */
        @keyframes fadeInDown {
          from { opacity: 0; transform: translateY(-28px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        /* ─── NAVBAR ENTRANCE ─── */
        .lp-navbar {
          animation: fadeInDown 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        /* ─── HERO ENTRANCE ─── */
        .lp-hero-spheres {
          animation: spheresFadeIn 1s cubic-bezier(0.22, 1, 0.36, 1) 0.1s both;
        }

        @keyframes spheresFadeIn {
          from { opacity: 0; transform: translateX(-50%) translateY(-28px); }
          to   { opacity: 1; transform: translateX(-50%) translateY(0); }
        }

        .lp-hero-notes {
          animation: fadeInDown 1s cubic-bezier(0.22, 1, 0.36, 1) 0.2s both;
        }

        .lp-hero-unleash-row {
          animation: fadeInDown 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.2s both;
        }

        .lp-hero-potential {
          animation: fadeInDown 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.35s both;
        }

        .lp-hero-subtitle {
          animation: fadeInDown 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.5s both;
        }

        .lp-hero-cta {
          animation: fadeInDown 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.65s both;
        }

        /* ─── SCROLL REVEAL (fade-in-down direction) ─── */
        .reveal {
          opacity: 0;
          transform: translateY(-28px);
          transition: opacity 0.75s cubic-bezier(0.22, 1, 0.36, 1),
                      transform 0.75s cubic-bezier(0.22, 1, 0.36, 1);
        }

        .reveal.revealed {
          opacity: 1;
          transform: translateY(0);
        }

        /* Stagger direct children of a reveal container */
        .reveal-stagger > * {
          opacity: 0;
          transform: translateY(-22px);
          transition: opacity 0.65s cubic-bezier(0.22, 1, 0.36, 1),
                      transform 0.65s cubic-bezier(0.22, 1, 0.36, 1);
        }

        .reveal-stagger.revealed > *:nth-child(1) { transition-delay: 0s;    opacity: 1; transform: none; }
        .reveal-stagger.revealed > *:nth-child(2) { transition-delay: 0.1s;  opacity: 1; transform: none; }
        .reveal-stagger.revealed > *:nth-child(3) { transition-delay: 0.2s;  opacity: 1; transform: none; }
        .reveal-stagger.revealed > *:nth-child(4) { transition-delay: 0.3s;  opacity: 1; transform: none; }

        /* ─── RESPONSIVE ─── */

        /* Large tablet */
        @media (max-width: 1200px) {
          .landing-root { padding: 24px; gap: 56px; }
          .lp-hero-wave-yellow { left: -24px; }
          .lp-nav-links { gap: 32px; }
          .lp-export-badges { display: none; }
          .lp-hero-device { width: 600px; height: 348px; }
          .lp-hero-visuals { height: 360px; }
          .lp-create-body { flex-direction: column; gap: 40px; }
          .lp-genre-visual { width: 100%; height: 0; }
          .lp-genre-visual-bg1, .lp-genre-visual-bg2 { display: none; }
          .lp-create-right { width: 100%; gap: 40px; }
          .lp-bento-row { flex-wrap: wrap; }
          .lp-bento-card.w480, .lp-bento-card.w480sq, .lp-bento-card.w642 {
            width: calc(50% - 10px);
            height: auto;
            min-height: 320px;
          }
          .lp-founder-section { gap: 40px; }
          .lp-founder-byline { max-width: 100%; flex-shrink: 1; }
        }

        /* Tablet */
        @media (max-width: 1024px) {
          .landing-root { gap: 48px; }
          .lp-hero-device { width: 480px; height: 278px; }
          .lp-hero-visuals { height: 300px; }
          .lp-bento-row { flex-direction: column; }
          .lp-bento-card.flex-1,
          .lp-bento-card.w480,
          .lp-bento-card.w480sq,
          .lp-bento-card.w642 { width: 100%; height: auto; min-height: 280px; }
          .lp-founder-section { flex-direction: column; gap: 32px; align-items: flex-start; }
          .lp-founder-byline { max-width: 100%; }
          .lp-founder-card { flex-shrink: 1; width: 100%; }
          .lp-demo-section { height: 320px; }
        }

        /* Mobile */
        @media (max-width: 768px) {
          .landing-root { padding: 16px; gap: 40px; }
          .lp-hero-wave-yellow { left: -16px; }
          .lp-nav-links { display: none; }
          .lp-nav-signin { font-size: 15px; padding: 10px 20px; }
          .lp-hero { min-height: auto; }
          .lp-hero-spheres { width: min(85vw, 480px); height: auto; }
          .lp-hero-notes img { width: 320px; height: 249px; }
          .lp-hero-notes img:first-child { left: -20px; }
          .lp-hero-notes img:last-child { left: 30vw; }
          .lp-hero-text { gap: 20px; }
          .lp-hero-unleash { white-space: normal; }
          .lp-hero-potential { white-space: normal; }
          .lp-hero-device { width: 88%; height: auto; }
          .lp-hero-visuals { height: 260px; margin-top: -12px; }
          .lp-export-badges { display: none; }
          .lp-audio-player { padding: 8px 16px; gap: 16px; }
          .lp-audio-title { font-size: 15px; }
          .lp-demo-section { height: 260px; }
          .lp-demo-play-icon { width: 110px; height: 110px; }
          .lp-bento-title { font-size: 18px; }
          .lp-bento-desc { font-size: 13px; }
          .lp-bento-ui-emotion { width: 260px; height: 230px; bottom: -32px; right: -24px; }
          .lp-bento-ui-collab { width: 260px; height: 230px; bottom: -32px; left: -24px; }
          .lp-founder-card { padding: 24px; gap: 20px; }
          .lp-founder-photo { width: 120px; height: 150px; }
          .lp-founder-info { gap: 24px; }
          .lp-founder-text { white-space: normal; }
          .lp-founder-role { font-size: 28px; }
          .lp-founder-name { font-size: 24px; }
          .lp-founder-title { font-size: 16px; }
          .lp-genres-grid { max-width: 100%; }
          .lp-footer { flex-direction: column; gap: 16px; align-items: flex-start; }
          .lp-footer-links { flex-wrap: wrap; }
        }

        /* Small mobile */
        @media (max-width: 480px) {
          .landing-root { padding: 12px; gap: 32px; }
          .lp-hero-wave-yellow { left: -12px; }
          .lp-nav-signin { font-size: 14px; padding: 9px 16px; }
          .lp-hero-visuals { height: 200px; margin-top: -8px; }
          .lp-hero-notes img { width: 200px; height: 156px; }
          .lp-hero-notes img:first-child { left: -10px; top: -20px; }
          .lp-hero-notes img:last-child { left: 25vw; top: 60px; }
          .lp-hero-spheres { width: 95vw; }
          .lp-hero-text { gap: 16px; }
          .lp-hero-unleash-row { gap: 8px; }
          .lp-hero-dash { width: 14px; height: 6px; }
          .lp-hero-subtitle { padding: 0 8px; }
          .lp-hero-device { width: 92%; }
          .lp-genre-pill { padding: 8px 14px; font-size: 12px; }
          .lp-audio-player { flex-direction: column; align-items: flex-start; gap: 12px; padding: 14px 16px; border-radius: 20px; }
          .lp-audio-waveform { display: none; }
          .lp-play-btn { width: 48px; height: 48px; align-self: center; }
          .lp-audio-info { align-items: flex-start; }
          .lp-demo-section { height: 200px; border-radius: 20px; }
          .lp-demo-play-icon { width: 80px; height: 80px; }
          .lp-bento-card.flex-1, .lp-bento-card.w480,
          .lp-bento-card.w480sq, .lp-bento-card.w642 { min-height: 220px; }
          .lp-bento-ui-emotion, .lp-bento-ui-collab { display: none; }
          .lp-key-box { width: 80px; height: 80px; font-size: 48px; }
          .lp-founder-card { padding: 16px; flex-direction: column; align-items: flex-start; }
          .lp-founder-photo { width: 90px; height: 112px; border-radius: 18px; }
          .lp-founder-info { flex-direction: column; gap: 16px; }
          .lp-founder-socials { flex-direction: row; }
          .lp-founder-role { font-size: 22px; }
          .lp-founder-name { font-size: 19px; }
          .lp-testimonial-section { padding: 32px 0; }
          .lp-footer { padding: 20px 16px; }
          .lp-footer-links { gap: 10px; }
          .lp-footer-copy { font-size: 12px; }
        }

        /* ── COMPATIBLE WORKSTATIONS ── */
        .lp-workstations-section {
          width: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 36px;
          padding: 32px 0;
        }

        .lp-ws-heading {
          font-size: clamp(16px, 1.9vw, 30px);
          font-weight: 500;
          letter-spacing: -0.02em;
          text-align: center;
          line-height: 1;
        }

        .lp-ws-heading .ws-dim   { color: #eaeaea; }
        .lp-ws-heading .ws-bright { color: #ffffff; }

        .lp-ws-container {
          width: 100%;
          overflow: hidden;
          position: relative;
          padding: 24px 0;
        }

        /* Fade edges into page background */
        .lp-ws-container::before,
        .lp-ws-container::after {
          content: '';
          position: absolute;
          top: 0;
          bottom: 0;
          width: 180px;
          z-index: 2;
          pointer-events: none;
        }
        .lp-ws-container::before { left: 0;  background: linear-gradient(to right, #0000ff 30%, transparent); }
        .lp-ws-container::after  { right: 0; background: linear-gradient(to left,  #0000ff 30%, transparent); }

        .lp-ws-track {
          display: flex;
          align-items: center;
          gap: 48px;
          width: max-content;
          transition: transform 0.65s cubic-bezier(0.4, 0, 0.2, 1);
          will-change: transform;
        }

        .lp-ws-logo {
          width: 160px;
          height: 150px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          flex-shrink: 0;
          transition: transform 0.65s cubic-bezier(0.4, 0, 0.2, 1),
                      opacity 0.65s ease;
          transform: scale(0.72);
          opacity: 0.35;
          user-select: none;
        }

        .lp-ws-logo img {
          width: 110px;
          height: 110px;
          object-fit: contain;
          display: block;
        }

        .lp-ws-logo.active {
          transform: scale(1.2);
          opacity: 1;
          animation: wsFocus 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }

        @keyframes wsFocus {
          0%   { transform: scale(0.72) rotate(-6deg); opacity: 0.35; }
          65%  { transform: scale(1.28) rotate(2deg);  opacity: 1; }
          100% { transform: scale(1.2)  rotate(0deg);  opacity: 1; }
        }

        @media (max-width: 768px) {
          .lp-ws-container::before,
          .lp-ws-container::after { width: 80px; }
          .lp-ws-logo { width: 120px; height: 110px; }
        }
      `}</style>

      {/* ── NAVBAR ── */}
      <nav className="lp-navbar">
        {/* Logo: vertical bars */}
        <Link to="/" className="lp-logo" aria-label="Arpeggiate home">
          <img src={imgLogo} alt="Arpeggiate" />
        </Link>

        <div className="lp-nav-links">
          <a href="#create" className="lp-nav-link">Explore</a>
          <a href="#create" className="lp-nav-link">Create</a>
          <a href="#why" className="lp-nav-link">About</a>
        </div>

        <Link to="/login" className="lp-nav-signin">Sign in</Link>
      </nav>

      {/* ── HERO ── */}
      <section className="lp-hero">
        {/* Background musical notes */}
        <div className="lp-hero-notes" ref={notesRef}>
          <img src={imgMusicalNotes} alt="" />
          <img src={imgMusicalNotes} alt="" />
        </div>

        {/* Spheres behind headline */}
        <img src={imgHeroFrame} alt="" className="lp-hero-spheres" ref={spheresRef} />

        {/* Headline */}
        <div className="lp-hero-text" ref={heroTextRef}>
          <div className="lp-hero-headline">
            <div className="lp-hero-unleash-row">
              <div className="lp-hero-dash" />
              <span className="lp-hero-unleash">UNLEASH</span>
              <div className="lp-hero-dash" />
            </div>
            <span className="lp-hero-potential">your potential</span>
          </div>

          <div className="lp-hero-sub">
            <p className="lp-hero-subtitle">
              Experience the Future of{' '}
              <span>Music Creation.</span>
            </p>
            <Link to="/login" className="lp-hero-cta">
              Start Building →
            </Link>
          </div>
        </div>

        {/* Hero visuals: waves + device, all relative to this container */}
        <div className="lp-hero-visuals" ref={visualsRef}>
          <img src={imgHeroWaveYellow} alt="" className="lp-hero-wave-yellow" />
          <img
            src={imgHeroDevice}
            alt="Music device"
            className="lp-hero-device"
          />

          {/* Cycling export format */}
          <div className="lp-export-badges">
            <span className="lp-export-badge">
              export as{' '}
              <em key={formatIdx} className="lp-export-format">
                {exportFormats[formatIdx]}
              </em>
            </span>
          </div>
        </div>
      </section>

      {/* ── COMPATIBLE WORKSTATIONS ── */}
      <section className="lp-workstations-section reveal">
        <h2 className="lp-ws-heading">
          <span className="ws-dim">Compatible </span>
          <span className="ws-bright">Workstations</span>
        </h2>
        <div className="lp-ws-container" ref={wsContainerRef}>
          <div className="lp-ws-track" ref={wsTrackRef}>
            {wsItems.map((ws, i) => (
              <div
                key={`${ws.name}-${i}`}
                className={`lp-ws-logo${i === activeWs ? ' active' : ''}`}
                onClick={() => setActiveWs(i)}
              >
                <img src={ws.img} alt={ws.name} />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CREATE WITH ARPEGGIATE ── */}
      <section className="lp-create-section reveal" id="create">
        <p className="lp-create-heading">
          Create with <em>ARPEGGIATE.</em>
        </p>

        <div className="lp-create-body">
          {/* Left: stacked card visual */}
          <div className="lp-genre-visual">
            <div className="lp-genre-visual-bg1" />
            <div className="lp-genre-visual-bg2" />
          </div>

          {/* Right: genre pills + audio player */}
          <div className="lp-create-right">
            <div className="lp-genres-grid">
              {genres.map(genre => (
                <button
                  key={genre}
                  className={`lp-genre-pill${selectedGenres.includes(genre) ? ' selected' : ''}`}
                  onClick={() => toggleGenre(genre)}
                >
                  {genre}
                </button>
              ))}
            </div>

            {/* Audio player bar */}
            <div className="lp-audio-player">
              <div className="lp-audio-info">
                <span className="lp-audio-title">LIFT.mp4  -  A.B.L</span>
                <div className="lp-audio-line">
                  <img src={imgAudioLine} alt="" />
                </div>
              </div>
              <div className="lp-play-btn">
                <img src={imgPlayButton} alt="Play" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── VIDEO / DEMO SECTION ── */}
      <section className="lp-demo-section reveal">
        <img src={imgPlaybackIcon} alt="Play demo" className="lp-demo-play-icon" />
      </section>

      {/* ── WHY CHOOSE ARPEGGIATE ── */}
      <section className="lp-why-section reveal" id="why">
        <h2 className="lp-why-heading">
          <span>Why Choose </span>Arpeggiate
        </h2>

        <div className="lp-bento-grid">
          {/* Row 1 */}
          <div className="lp-bento-row reveal-stagger reveal">
            {/* Bento 1: Emotion-driven Patterns */}
            <div className="lp-bento-card flex-1">
              <div className="lp-bento-text">
                <p className="lp-bento-title">Emotion-driven Patterns</p>
                <p className="lp-bento-desc">Generate musical patterns based on emotional input.</p>
              </div>
              <div className="lp-bento-img-area">
                <div className="lp-bento-ui-emotion">
                  <div className="lp-emotion-user-row">
                    <img src={imgMusicianPhoto} alt="" className="lp-emotion-avatar" />
                    <span className="lp-emotion-label">Emotion Analysis</span>
                  </div>
                  <div className="lp-emotion-bars">
                    <div className="lp-emotion-bar active" style={{ width: '75%' }} />
                    <div className="lp-emotion-bar" style={{ width: '63%' }} />
                    <div className="lp-emotion-bar" style={{ width: '70%' }} />
                    <div className="lp-emotion-bar" style={{ width: '75%' }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Bento 2: Producer's Block */}
            <div className="lp-bento-card w480">
              <div className="lp-bento-text">
                <p className="lp-bento-title">Overcome "Producer's Block"</p>
                <p className="lp-bento-desc">Generate arpeggios and easily export to continue iterating</p>
              </div>
              <div className="lp-bento-img-area" />
            </div>
          </div>

          {/* Row 2 */}
          <div className="lp-bento-row reveal-stagger reveal">
            {/* Bento 3: User-Friendly Interface */}
            <div className="lp-bento-card w480sq">
              <div className="lp-bento-text">
                <p className="lp-bento-title">User-Friendly Interface</p>
                <p className="lp-bento-desc">Easy-to-use interface for seamless music creation.</p>
              </div>
              <div className="lp-bento-img-area">
                <div className="lp-bento-ui-keys">
                  <div className="lp-key-box">⌘</div>
                  <div className="lp-key-box">U</div>
                </div>
              </div>
            </div>

            {/* Bento 4: Collaborative Features */}
            <div className="lp-bento-card w642">
              <div className="lp-bento-text">
                <p className="lp-bento-title">Collaborative Features</p>
                <p className="lp-bento-desc">Share and collaborate with other musicians.</p>
              </div>
              <div className="lp-bento-img-area">
                <div className="lp-bento-ui-collab">
                  <div className="lp-collab-buttons">
                    <div className="lp-collab-btn-ghost" style={{ width: '105px' }} />
                    <div className="lp-collab-btn-ghost" style={{ width: '80px' }} />
                    <div className="lp-collab-btn-solid">Collaboration</div>
                  </div>
                  <div className="lp-collab-content-bg" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── FOUNDER SECTION ── */}
      <section className="lp-founder-section reveal-stagger reveal">
        {/* Founder card */}
        <div className="lp-founder-card">
          <img
            src={imgFounderPhoto}
            alt="Abel Shegere"
            className="lp-founder-photo"
          />
          <div className="lp-founder-info">
            <div className="lp-founder-text">
              <span className="lp-founder-role">Founder</span>
              <span className="lp-founder-name">Abel Shegere</span>
              <span className="lp-founder-title">SWE &amp; Producer</span>
            </div>
            <div className="lp-founder-socials">
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="lp-social-icon-box"
                aria-label="Twitter"
              >
                <img src={imgIconTwitter} alt="Twitter" />
              </a>
              <a
                href="https://soundcloud.com"
                target="_blank"
                rel="noopener noreferrer"
                className="lp-social-icon-box"
                aria-label="SoundCloud"
              >
                {/* Inline SoundCloud icon using SVG dots */}
                <svg viewBox="0 0 32 32" width="30" height="30" fill="white" xmlns="http://www.w3.org/2000/svg">
                  <path d="M1 20c0 2.76 2.24 5 5 5h17c2.76 0 5-2.24 5-5s-2.24-5-5-5c-.34 0-.68.03-1 .1C21.4 11.6 18.5 9 15 9c-3.87 0-7 3.13-7 7 0 .09.01.18.01.27C6.42 16.73 4 18.18 4 20H1z"/>
                  <path d="M0 21c0 1.1.9 2 2 2s2-.9 2-2-.9-2-2-2-2 .9-2 2z" opacity="0.5"/>
                </svg>
              </a>
            </div>
          </div>
        </div>

        {/* By producers for producers */}
        <div className="lp-founder-byline">
          <h2 className="lp-founder-headline">
            By&nbsp; producers,{'\n'}for producers
          </h2>
          <p className="lp-founder-desc">
            Our user-friendly interface designed by Carnegie Mellon alumni ensures a seamless experience, allowing you to focus on creativity.
          </p>
        </div>
      </section>

      {/* ── TESTIMONIAL ── */}
      <section className="lp-testimonial-section reveal">
        <p className="lp-testimonial-quote">
          The emotional analysis feature is truly groundbreaking and allows me to spend more time editing than brainstorming sequences.
        </p>
        <div className="lp-testimonial-author">
          <img src={imgUserAvatar} alt="GJ Jaris" className="lp-testimonial-avatar" />
          <div className="lp-testimonial-author-info">
            <div className="lp-testimonial-brand-row">
              <img src={imgLogomark} alt="" className="lp-testimonial-brand-logo" />
              <span className="lp-testimonial-brand-name">JARIS</span>
            </div>
            <span className="lp-testimonial-role">Music Producer</span>
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="lp-footer">
        <span className="lp-footer-copy">© 2025 Arpeggiate.ai - Powered by AI</span>
        <div className="lp-footer-links">
          <a href="#" className="lp-footer-link-pill">
            <img src={imgIconCoffee} alt="" />
            <span>Buy me a coffee</span>
          </a>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="lp-footer-icon-link"
            aria-label="GitHub"
          >
            <img src={imgIconGithub} alt="GitHub" />
          </a>
          <a
            href="https://twitter.com"
            target="_blank"
            rel="noopener noreferrer"
            className="lp-footer-icon-link"
            aria-label="Twitter"
          >
            <img src={imgIconTwitter} alt="Twitter" />
          </a>
          <a
            href="https://www.linkedin.com/in/abelshegere"
            target="_blank"
            rel="noopener noreferrer"
            className="lp-footer-icon-link"
            aria-label="LinkedIn"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" fill="#cad5e2"/>
              <rect x="2" y="9" width="4" height="12" fill="#cad5e2"/>
              <circle cx="4" cy="4" r="2" fill="#cad5e2"/>
            </svg>
          </a>
        </div>
      </footer>
    </div>
  );
}
