import React from 'react';
import { Hash } from 'lucide-react';
import './Trusted.css';

const Trusted = () => {
  const brands = [
    'LOGIC',
    'ABLETON',
    'FL STUDIO',
    'KONTAKT',
    'SERUM'
  ];

  return (
    <section className="trusted-section">
      <div className="container">
        <h2 className="trusted-title">Trusted by Leading Music Producers</h2>
        <div className="trusted-logos">
          {brands.map((brand, index) => (
            <div key={index} className="logo-item">
              <Hash size={20} />
              {brand}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Trusted;
