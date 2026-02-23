import React from 'react';
import { Music } from 'lucide-react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">
              <Music size={18} strokeWidth={3} />
            </div>
            arpeggiator.ai
          </div>
          <nav className="nav">
            <Link to="/" className="nav-link">Home</Link>
            <Link to="/login" className="nav-link">Sign In</Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
