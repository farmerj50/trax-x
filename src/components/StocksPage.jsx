import React, { useState, useEffect } from "react";
import ErrorBoundary from "./ErrorBoundary";
import StockTracker from "./StockTracker";
import io from "socket.io-client";
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

const socket = io("http://localhost:5000");

const StocksPage = () => {
  const [ticker, setTicker] = useState("");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [chartData, setChartData] = useState([]); // Ensure chartData defaults to an empty array
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const periods = [
    { intervalType: "Months", interval: 1, text: "1M" },
    { intervalType: "Months", interval: 3, text: "3M" },
    { intervalType: "Months", interval: 6, text: "6M" },
    { intervalType: "Years", interval: 1, text: "YTD" },
    { intervalType: "Years", interval: 3, text: "All" },
  ];

  // Fetch historical chart data
  const fetchChartData = async (tickerSymbol) => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:5000/api/candlestick?ticker=${tickerSymbol}`
      );
      if (!response.ok) throw new Error("Failed to fetch chart data.");

      const data = await response.json();
      if (data && data.dates && data.dates.length > 0) {
        const formattedData = data.dates
          .map((date, index) => ({
            x: new Date(date),
            open: data.open[index],
            high: data.high[index],
            low: data.low[index],
            close: data.close[index],
          }))
          .filter(
            (entry) =>
              entry.x &&
              !isNaN(entry.open) &&
              !isNaN(entry.high) &&
              !isNaN(entry.low) &&
              !isNaN(entry.close)
          );

        setChartData(formattedData);
        setError("");
      } else {
        setChartData([]); // Ensure chartData is reset if no data is available
        setError("No data available for the selected ticker.");
      }
    } catch (err) {
      console.error("Error fetching chart data:", err);
      setChartData([]);
      setError(err.message || "Failed to load chart data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChartData(selectedTicker);
  }, [selectedTicker]);

  useEffect(() => {
    console.log("🔄 Chart Data Updated:", chartData);
    if (!chartData || chartData.length === 0) {
      console.warn("⚠ Chart Data is empty. Avoiding Syncfusion render.");
    }
  }, [chartData]);

  const handleSearch = () => {
    if (ticker.trim() !== "") {
      setSelectedTicker(ticker.toUpperCase());
    }
  };

  return (
    <div className="stocks-page" style={{ padding: "20px" }}>
      <div className="stock-header">
        <h2 style={{ textAlign: "center" }}>{selectedTicker} Stock Analysis</h2>
      </div>

      <div className="search-stock">
        <input
          type="text"
          placeholder="Enter stock ticker (e.g., AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
        />
        <button onClick={handleSearch}>Search</button>
      </div>

      <div id="chart-container">
        {loading ? (
          <p style={{ textAlign: "center", color: "gray" }}>Loading...</p>
        ) : error ? (
          <p style={{ color: "red", textAlign: "center" }}>{error}</p>
        ) : chartData && chartData.length > 0 ? ( // Ensure chartData is not null
          <ErrorBoundary>
            <StockChartComponent
              id="stockchart"
              enableSelector={true}
              primaryXAxis={{
                valueType: "DateTime",
                labelFormat: "MMM dd",
                majorGridLines: { width: 0 },
                intervalType: "Days",
                crosshairTooltip: { enable: true },
              }}
              primaryYAxis={{
                labelFormat: "${value}",
                majorGridLines: { width: 0 },
                rangePadding: "None",
                crosshairTooltip: { enable: true },
              }}
              tooltip={{ enable: true }}
              crosshair={{ enable: true }}
              periods={periods}
              title={`${selectedTicker} Stock Analysis`}
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
          </ErrorBoundary>
        ) : (
          <p style={{ textAlign: "center", color: "gray" }}>No data available.</p>
        )}
      </div>
    </div>
  );
};

export default StocksPage;
