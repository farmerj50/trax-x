import React from "react";
import CandlestickChart from "./CandlestickChart";
import "./StockScanner.css";

const StockScanner = ({ stocks }) => {
  return (
    <div className="stock-scanner">
      {/* <h2>Stock Results</h2> */}

      {/* Display Stock Results */}
      <div className="stock-list">
        {stocks && stocks.length > 0 ? (
          stocks.map((stock) => (
            <div className="stock-card" key={stock.T}>
              <CandlestickChart
                ticker={stock.T} // Pass ticker symbol
                entryPoint={stock.c * 0.95} // Entry point is 5% below close
                exitPoint={stock.c * 1.1} // Exit point is 10% above close
                additionalData={{
                  volatility: `${(stock.volatility * 100).toFixed(2)}%`,
                  change: `${(stock.price_change * 100).toFixed(2)}%`,
                  rsi: stock.rsi || "N/A",
                }} // Pass additional data
              />
            </div>
          ))
        ) : (
          <p style={{ textAlign: "center" }}>
            No stocks found. Please adjust your criteria.
          </p>
        )}
      </div>
    </div>
  );
};

export default StockScanner;
