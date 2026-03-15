import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          {/* Logo slot */}
        </div>

        <nav className="nav-pill">
          <Link to="/explore" className="nav-pill-link">Explore</Link>
          <Link to="/create" className="nav-pill-link">Create</Link>
          <Link to="/about" className="nav-pill-link">About</Link>
        </nav>

        <div className="navbar-actions">
          <Link to="/login" className="signin-btn">Sign in</Link>
        </div>
      </div>
    </header>
  );
}
