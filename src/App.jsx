import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Navbar from "./components/Navbar";
import StockScanner from "./components/StockScanner";
import SearchForm from "./components/SearchForm";
import TickerNewsWidget from "./components/TickerNewsWidget";
import StocksPage from "./components/StocksPage";
import OptionsPage from "./components/OptionsPage";
import CryptoPage from "./components/CryptoPage";
import ShortSalesPage from "./components/ShortSalesPage";
import "./App.css";

const App = () => {
  const [stocks, setStocks] = useState([]);
  const [tickers, setTickers] = useState([]);

  // Dark Mode State: Use localStorage or system preference
  const storedTheme = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = storedTheme || (prefersDark ? "dark" : "light");

  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    // Apply dark mode class to the body
    document.body.classList.toggle("dark-mode", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

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
    <Router>
      {/* Navigation Menu */}
      <div className={`menu-bar ${theme}`}>
        <h1 className="menu-title">AI Stock Scanner</h1>
        <div className="menu-buttons">
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light Mode" : "Dark Mode"}
          </button>
          <Link to="/"><button>Home</button></Link>
          <Link to="/stocks"><button>Stocks</button></Link>
          <Link to="/options"><button>Options</button></Link>
          <Link to="/crypto"><button>Crypto</button></Link>
          <Link to="/short-sales"><button>Short Sales</button></Link>
        </div>
      </div>

      {/* Define Routes */}
      <Routes>
        <Route
          path="/"
          element={
            <div className={`app-layout ${theme}`}>
              {/* Search Form */}
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
          }
        />
        <Route path="/stocks" element={<StocksPage />} />
        <Route path="/options" element={<OptionsPage />} />
        <Route path="/crypto" element={<CryptoPage />} />
        <Route path="/short-sales" element={<ShortSalesPage />} />
      </Routes>
    </Router>
  );
};

export default App;
