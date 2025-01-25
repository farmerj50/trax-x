import React, { useState, useEffect } from "react";
import ErrorBoundary from "./ErrorBoundary"; // Import the ErrorBoundary component
import "./StocksPage.css";

import {
  StockChartComponent,
  StockChartSeriesCollectionDirective,
  StockChartSeriesDirective,
  Inject,
  DateTime,
  Tooltip,
  RangeTooltip,
  Crosshair,
  LineSeries,
  CandleSeries,
  Legend,
  Export,
  EmaIndicator,
  TmaIndicator,
  SmaIndicator,
  MomentumIndicator,
  AtrIndicator,
  AccumulationDistributionIndicator,
  BollingerBands,
  MacdIndicator,
  StochasticIndicator,
  RsiIndicator,
} from "@syncfusion/ej2-react-charts";

const StocksPage = () => {
  const [ticker, setTicker] = useState(""); // Input for ticker search
  const [selectedTicker, setSelectedTicker] = useState("AAPL"); // Default ticker
  const [chartData, setChartData] = useState([]); // Holds chart data
  const [error, setError] = useState(""); // Tracks errors during data fetching
  const [selectorOptions] = useState(["1M", "3M", "6M", "1Y", "YTD", "All"]);

  const fetchChartData = async (tickerSymbol) => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/candlestick?ticker=${tickerSymbol}`
      );
      const data = await response.json();
      if (data.dates) {
        const formattedData = data.dates.map((date, index) => ({
          x: new Date(date),
          open: data.open[index],
          high: data.high[index],
          low: data.low[index],
          close: data.close[index],
        }));
        setChartData(formattedData);
        setError("");
      } else {
        setChartData([]);
        setError("No data available for the selected ticker.");
      }
    } catch (err) {
      console.error("Error fetching chart data:", err);
      setChartData([]); // Reset chart data in case of an error
      setError("Failed to load chart data.");
    }
  };

  useEffect(() => {
    fetchChartData(selectedTicker);
  }, [selectedTicker]);

  const handleSearch = () => {
    if (ticker.trim() !== "") {
      setSelectedTicker(ticker.toUpperCase());
    }
  };

  return (
    <div className="stocks-page" style={{ padding: "20px" }}>
      {/* Header Section */}
      <div className="stock-header" style={{ marginBottom: "20px" }}>
        <h2 style={{ textAlign: "center", fontSize: "24px" }}>
          {selectedTicker} Stock Analysis
        </h2>
      </div>

      {/* Search Field */}
      <div
        className="search-stock"
        style={{
          marginBottom: "20px",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <input
          type="text"
          placeholder="Enter stock ticker (e.g., AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          style={{
            padding: "10px",
            fontSize: "16px",
            borderRadius: "5px",
            border: "1px solid #ccc",
          }}
        />
        <button
          onClick={handleSearch}
          style={{
            padding: "10px 20px",
            fontSize: "16px",
            borderRadius: "5px",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            cursor: "pointer",
          }}
        >
          Search
        </button>
      </div>

      {/* Chart Section */}
      <div
        id="chart-container"
        style={{
          width: "100%",
          height: "500px",
          backgroundColor: "#fff",
          borderRadius: "8px",
          padding: "10px",
        }}
      >
        {error ? (
          <p style={{ color: "red", textAlign: "center" }}>{error}</p>
        ) : (
          <ErrorBoundary>
            {chartData.length > 0 ? (
              <StockChartComponent
                id="stockchart"
                primaryXAxis={{
                  valueType: "DateTime",
                  labelFormat: "MMM dd",
                  majorGridLines: { width: 0 },
                  intervalType: "Days",
                }}
                primaryYAxis={{
                  labelFormat: "${value}",
                  majorGridLines: { width: 0 },
                  rangePadding: "None",
                }}
                legendSettings={{ visible: true }}
                title={`${selectedTicker} Stock Analysis`}
                enableSelector={true}
                selectorSettings={{
                  items: selectorOptions,
                }}
                height="100%"
                width="100%"
              >
                <Inject
                  services={[
                    DateTime,
                    Tooltip,
                    RangeTooltip,
                    Crosshair,
                    LineSeries,
                    CandleSeries,
                    Legend,
                    Export,
                    EmaIndicator,
                    TmaIndicator,
                    SmaIndicator,
                    MomentumIndicator,
                    AtrIndicator,
                    AccumulationDistributionIndicator,
                    BollingerBands,
                    MacdIndicator,
                    StochasticIndicator,
                    RsiIndicator,
                  ]}
                />
                <StockChartSeriesCollectionDirective>
                  <StockChartSeriesDirective
                    dataSource={chartData}
                    xName="x"
                    open="open"
                    high="high"
                    low="low"
                    close="close"
                    type="Candle"
                    animation={{ enable: true }}
                  />
                </StockChartSeriesCollectionDirective>
              </StockChartComponent>
            ) : (
              <p style={{ textAlign: "center", color: "gray" }}>
                Loading chart data, please wait...
              </p>
            )}
          </ErrorBoundary>
        )}
      </div>
    </div>
  );
};

export default StocksPage;
