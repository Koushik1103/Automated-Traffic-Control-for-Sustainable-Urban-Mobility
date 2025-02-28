import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Analytics.css";

const Analytics = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  const handleSearch = () => {
    fetch(`http://localhost:5000/get_traffic_data`)
      .then((response) => response.json())
      .then((data) => {
        const results = data.top_3_congested_locations.filter((location) =>
          location.location.toLowerCase().includes(searchQuery.toLowerCase())
        );
        setSearchResults(results);
      })
      .catch((error) => console.error("Error fetching search results:", error));
  };

  return (
    <div className="analytics-container">
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

      <div className="search-container">
        <div className="search-box">
          <input
            type="text"
            placeholder="🔍 Search by location"
            className="search-input"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button onClick={handleSearch} className="search-btn">
            Search
          </button>
          <div className="analytics-card">
            {searchResults.length > 0 ? (
              searchResults.map((result, index) => (
                <p key={index}>
                  {result.location} - Score: {result.congestion_score}
                </p>
              ))
            ) : (
              <p>No results found.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;

// import React from "react";
// import { useNavigate } from "react-router-dom";
// import "./Analytics.css";

// const Analytics = () => {
//   const navigate = useNavigate();

//   return (
//     <div className="analytics-container">
//       <nav className="navbar">
//         <button className="nav-btn" onClick={() => navigate("/home")}>Home</button>
//         <button className="nav-btn" onClick={() => navigate("/admin")}>Admin</button>
//       </nav>

//       <div className="search-container">
//         <div className="search-box">
//           <input type="text" placeholder="🔍 Search" className="search-input" />
//           <div className="analytics-card">
//             <p>Lorem ipsum dolor sit amet.</p>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Analytics;
