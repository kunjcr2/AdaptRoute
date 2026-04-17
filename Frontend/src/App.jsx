import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Architecture from './pages/Architecture';
import Firewall from './pages/Firewall';
import Evaluation from './pages/Evaluation';
import Demo from './pages/Demo';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="architecture" element={<Architecture />} />
        <Route path="firewall" element={<Firewall />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="demo" element={<Demo />} />
      </Route>
    </Routes>
  );
}

export default App;