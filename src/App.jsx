import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ArpeggiatorLanding from './landing-page';
import Login from './components/Login';
import Signup from './components/Signup';
import Build from './components/Build';
import SavedMidis from './components/SavedMidis';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ArpeggiatorLanding />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/build" element={<Build />} />
        <Route path="/saved" element={<SavedMidis />} />
      </Routes>
    </Router>
  );
}

export default App;
