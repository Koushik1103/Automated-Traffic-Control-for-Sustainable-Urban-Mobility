import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Admin.css";

const Admin = () => {
  const navigate = useNavigate();
  const [topCongestedLocations, setTopCongestedLocations] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5000/get_traffic_data")
      .then((response) => response.json())
      .then((data) => setTopCongestedLocations(data.top_3_congested_locations))
      .catch((error) =>
        console.error("Error fetching congested locations:", error)
      );
  }, []);

  return (
    <div className="container">
      <nav className="navbar">
        <button className="nav-btn" onClick={() => navigate("/home")}>
          Home
        </button>
        <button className="nav-btn" onClick={() => navigate("/admin")}>
          Admin
        </button>
        <button className="nav-btn" onClick={() => navigate("/analytics")}>
          Analytics
        </button>
      </nav>
      <div className="congestion-card">
        <h2>Top 3 Congested Locations</h2>
        <ul>
          {topCongestedLocations.length > 0 ? (
            topCongestedLocations.map((location, index) => (
              <li key={index}>
                {location.location} - Score: {location.congestion_score}
              </li>
            ))
          ) : (
            <p>Loading data...</p>
          )}
        </ul>
      </div>
    </div>
  );
};

export default Admin;

// import React from "react";
// import { useNavigate } from "react-router-dom";
// import "./Admin.css";

// const Admin = () => {
//   const navigate = useNavigate();

//   return (
//     <div className="container">
//       <nav className="navbar">
//         <button className="nav-btn" onClick={() => navigate("/home")}>Home</button>
//         <button className="nav-btn" onClick={() => navigate("/admin")}>Admin</button>
//       </nav>
//     </div>
//   );
// };

// export default Admin;
