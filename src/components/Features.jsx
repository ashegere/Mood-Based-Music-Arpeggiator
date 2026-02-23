import React from 'react';
import FeatureCard from './FeatureCard';
import { Sparkles, Music, Users } from 'lucide-react';
import './Features.css';

const Features = () => {
  const features = [
    {
      title: 'Emotion-driven Patterns',
      description: 'Generate evolving patterns based on emotional input.',
      icon: <Sparkles size={48} style={{ color: '#ff6b00' }} />,
      type: 'large',
      hasImage: true
    },
    {
      title: 'Overcome "Producer\'s Block"',
      description: 'Generate a suggestive and totally original compositions to continue working on.',
      icon: <Music size={48} style={{ color: '#ff6b00' }} />,
      type: 'large',
      hasImage: true,
      gradient: true
    },
    {
      title: 'User-Friendly Interface',
      description: 'Easy-to-use interface for seamless music creation.',
      type: 'icons',
      icons: ['⌘', 'U']
    },
    {
      title: 'Collaborative Features',
      description: 'Share and collaborate with other musicians.',
      badge: 'Collaborative',
      icon: <Users size={40} style={{ color: 'rgba(255,255,255,0.3)' }} />,
      hasImage: true,
      imageHeight: '120px'
    }
  ];

  return (
    <section className="features-section">
      <div className="container">
        <h2 className="section-title">Why Choose ArpeggiateAI</h2>
        
        <div className="features-grid">
          {features.map((feature, index) => (
            <FeatureCard key={index} {...feature} />
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
