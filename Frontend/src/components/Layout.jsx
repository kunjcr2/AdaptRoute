import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import { Network } from 'lucide-react';

const Layout = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
