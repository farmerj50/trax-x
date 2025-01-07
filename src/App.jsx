import React, { useState } from "react";
import Navbar from "./components/Navbar";
import StockScanner from "./components/StockScanner";
import SearchForm from "./components/SearchForm";
import TickerNewsWidget from "./components/TickerNewsWidget";
import "./App.css";

const App = () => {
  const [stocks, setStocks] = useState([]);
  const [tickers, setTickers] = useState([]);

  const fetchStocks = async (criteria) => {
    try {
      const queryParams = new URLSearchParams({
        min_price: criteria.minPrice,
        max_price: criteria.maxPrice,
        min_rsi: criteria.minRSI,
        max_rsi: criteria.maxRSI,
        volume_surge: criteria.volumeSurge,
      });

      const response = await fetch(
        `http://localhost:5000/api/scan-stocks?${queryParams.toString()}`
      );

      if (!response.ok) {
        throw new Error("Failed to fetch stocks.");
      }

      const data = await response.json();

      if (data.candidates && data.candidates.length > 0) {
        setStocks(data.candidates);
        setTickers(data.candidates.map((stock) => stock.T));
      } else {
        setStocks([]);
        setTickers([]);
      }
    } catch (err) {
      console.error("Error fetching stocks:", err);
      setStocks([]);
      setTickers([]);
    }
  };

  return (
    <>
      <Navbar />
      <div className="app-layout">
        {/* Search Bar */}
        <div className="search-bar">
          <SearchForm onSearch={fetchStocks} />
        </div>

        {/* Stock Results Header */}
        <div className="stock-results-header">
          <h2>Stock Results</h2>
        </div>

        {/* Main Content */}
        <div className="main-content">
          <StockScanner stocks={stocks} />
          <div className="news-widget">
            <TickerNewsWidget tickers={tickers} />
          </div>
        </div>
      </div>
    </>
  );
};

export default App;
