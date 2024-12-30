import React, { useEffect, useState } from "react";
import CandlestickChart from "./CandlestickChart";
import "./StockScanner.css";

const StockScanner = () => {
  const [stocks, setStocks] = useState([]); // Initialize as an empty array
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [minRSI, setMinRSI] = useState(30); // Default RSI range
  const [maxRSI, setMaxRSI] = useState(70);
  const [volumeSurge, setVolumeSurge] = useState(1.2); // Default volume surge
  const [error, setError] = useState("");

  // Fetch stocks based on criteria
  const fetchStocks = async () => {
    try {
      const queryParams = new URLSearchParams({
        min_price: minPrice || "0",
        max_price: maxPrice || "1000000",
        min_rsi: minRSI,
        max_rsi: maxRSI,
        volume_surge: volumeSurge,
      });

      const response = await fetch(
        `http://localhost:5000/api/scan-stocks?${queryParams.toString()}`
      );
      const data = await response.json();

      if (response.ok) {
        if (data.candidates && data.candidates.length > 0) {
          setStocks(data.candidates); // Store the array of stocks
          setError("");
        } else {
          setStocks([]); // No stocks returned
          setError("No stocks found. Please adjust your criteria.");
        }
      } else {
        setError(data.error || "Failed to fetch stocks.");
      }
    } catch (err) {
      setError("Failed to fetch stocks.");
      console.error(err);
    }
  };

  return (
    <div className="stock-scanner">
      <h2 style={{ textAlign: "center" }}>Find Stocks by Custom Criteria</h2>

      {/* Form for filtering criteria */}
      <div className="filter-criteria">
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
        <button onClick={fetchStocks}>Search Stocks</button>
      </div>

      {/* Display error message */}
      {error && <p style={{ color: "red", textAlign: "center" }}>{error}</p>}

      {/* Display stock results */}
      <div className="stock-list">
        {stocks && stocks.length > 0 ? (
          stocks.map((stock) => (
            <CandlestickChart
              key={stock.T} // Use "T" for the ticker key
              ticker={stock.T} // Use "T" for the ticker
              entryPoint={stock.c * 0.95} // Use "c" (close price) for calculations
              exitPoint={stock.c * 1.10} // Use "c" (close price) for calculations
              additionalData={`Volatility: ${(stock.volatility * 100).toFixed(
                2
              )}%, Change: ${(stock.price_change * 100).toFixed(2)}%, RSI: ${
                stock.rsi
              }`}
            />
          ))
        ) : (
          <p style={{ textAlign: "center" }}>No stocks found. Please adjust your criteria.</p>
        )}
      </div>
    </div>
  );
};

export default StockScanner;
