import React from "react";
import { useNavigate } from "react-router-dom";
import "./Admin.css";

const Admin = () => {
  const navigate = useNavigate();

  return (
    <div className="container">
      <nav className="navbar">
        <button className="nav-btn">Home</button>
        <button className="nav-btn">Admin</button>
      </nav>
    </div>
  );
};

export default Admin;