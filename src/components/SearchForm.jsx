import React, { useState } from "react";
import "./SearchForm.css";

const SearchForm = ({ onSearch }) => {
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [minRSI, setMinRSI] = useState(30);
  const [maxRSI, setMaxRSI] = useState(70);
  const [volumeSurge, setVolumeSurge] = useState(1.2);

  const handleSearch = () => {
    // Create search parameters and pass to parent
    const criteria = {
      minPrice: minPrice || "0",
      maxPrice: maxPrice || "1000000",
      minRSI: minRSI || "0",
      maxRSI: maxRSI || "100",
      volumeSurge: volumeSurge || "1",
    };
    onSearch(criteria);
  };

  return (
    <div className="search-form">
      <label>
        Min Price:
        <input
          type="number"
          value={minPrice}
          onChange={(e) => setMinPrice(e.target.value)}
          placeholder="Enter min price"
        />
      </label>
      <label>
        Max Price:
        <input
          type="number"
          value={maxPrice}
          onChange={(e) => setMaxPrice(e.target.value)}
          placeholder="Enter max price"
        />
      </label>
      <label>
        Min RSI:
        <input
          type="number"
          value={minRSI}
          onChange={(e) => setMinRSI(e.target.value)}
          placeholder="Enter min RSI"
        />
      </label>
      <label>
        Max RSI:
        <input
          type="number"
          value={maxRSI}
          onChange={(e) => setMaxRSI(e.target.value)}
          placeholder="Enter max RSI"
        />
      </label>
      <label>
        Volume Surge:
        <input
          type="number"
          step="0.1"
          value={volumeSurge}
          onChange={(e) => setVolumeSurge(e.target.value)}
          placeholder="Enter volume surge"
        />
      </label>
      <button onClick={handleSearch}>Search Stocks</button>
    </div>
  );
};

export default SearchForm;
