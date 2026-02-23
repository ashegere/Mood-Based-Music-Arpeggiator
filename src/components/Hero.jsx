import React from 'react';
import { ChevronRight } from 'lucide-react';
import './Hero.css';

const Hero = () => {
  return (
    <section className="hero">
      <h1 className="hero-title">
        Make your music
        <span className="energetic">energetic.</span>
      </h1>

      <button className="cta-button">
        Start Building
        <ChevronRight size={20} />
      </button>
    </section>
  );
};

export default Hero;
