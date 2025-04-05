import React, { useState, useEffect } from "react";
import ErrorBoundary from "./ErrorBoundary";
import StockTracker from "./StockTracker";
import LiveStockUpdate from "./LiveStockUpdate";
import AddTicker from "./AddTicker";
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
  ScatterSeries
} from "@syncfusion/ej2-react-charts";

const POLYGON_WS_URL = "wss://delayed.polygon.io/stocks"; // 15-min delayed data
const POLYGON_API_KEY = process.env.REACT_APP_POLYGON_API_KEY;

const StocksPage = () => {
  const [ticker, setTicker] = useState("");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [chartData, setChartData] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [livePrice, setLivePrice] = useState(null);
  const [ws, setWs] = useState(null);
  const [entryPoint, setEntryPoint] = useState(null);
  const [exitPoint, setExitPoint] = useState(null);

  const periods = [
    { intervalType: "Months", interval: 1, text: "1M" },
    { intervalType: "Months", interval: 3, text: "3M" },
    { intervalType: "Months", interval: 6, text: "6M" },
    { intervalType: "Years", interval: 1, text: "YTD" },
    { intervalType: "Years", interval: 3, text: "All" },
  ];

  /** 🔄 Fetch Historical Chart Data */
  const fetchStockData = async (tickerSymbol) => {
    setLoading(true);
    console.log(`📡 Fetching stock data for: ${tickerSymbol}`);

    try {
      const response = await fetch(
        `http://localhost:5000/api/stock-data?ticker=${tickerSymbol}`
      );

      if (!response.ok) throw new Error("Failed to fetch stock data.");

      const data = await response.json();
      console.log("✅ API Response:", data);

      if (data && data.dates && data.dates.length > 0) {
        const formattedData = data.dates.map((date, index) => ({
          x: new Date(date),
          open: data.open[index],
          high: data.high[index],
          low: data.low[index],
          close: data.close[index],
        }));

        setChartData(formattedData);
        setEntryPoint(data.entry_point ? data.entry_point : null);
        setExitPoint(data.exit_point ? data.exit_point : null);
        setError("");
      } else {
        setChartData([]);
        setError("No data available for the selected ticker.");
      }
    } catch (err) {
      console.error("❌ Error fetching stock data:", err);
      setChartData([]);
      setError(err.message || "Failed to load stock data.");
    } finally {
      setLoading(false);
    }
  };

  /** 🔄 Handle Live WebSocket Updates */
  const handleWebSocketMessage = (event) => {
    const data = JSON.parse(event.data);

    data.forEach((update) => {
      if (update.ev === "AM" && update.sym === selectedTicker) {
        console.log("📡 Received WebSocket Update:", update);
        setLivePrice(update.c);

        // Update chart with latest minute aggregate close price
        setChartData((prevData) => {
          if (prevData.length === 0) return prevData;
          const newData = [...prevData];
          newData[newData.length - 1] = {
            ...newData[newData.length - 1],
            close: update.c, // Update only close price
          };
          return newData;
        });
      }
    });
  };

  /** 🔄 Setup WebSocket Connection */
  const setupWebSocket = () => {
    if (ws) {
      ws.close(); // Close existing connection
    }

    const websocket = new WebSocket(POLYGON_WS_URL);
    websocket.onopen = () => {
      console.log("✅ Connected to Polygon.io WebSocket");
      websocket.send(JSON.stringify({ action: "auth", params: POLYGON_API_KEY }));
      websocket.send(JSON.stringify({ action: "subscribe", params: `AM.${selectedTicker}` }));
    };

    websocket.onmessage = handleWebSocketMessage;
    websocket.onerror = (err) => console.error("❌ WebSocket Error:", err);
    websocket.onclose = () => console.log("⚠ WebSocket Disconnected");

    setWs(websocket);
  };

  /** 🔄 Effect: Fetch Chart Data on Ticker Change */
  useEffect(() => {
    fetchStockData(selectedTicker);
    setupWebSocket();
    return () => {
      if (ws) ws.close();
    };
  }, [selectedTicker]);

  /** 🔎 Handle Search */
  const handleSearch = () => {
    if (ticker.trim() !== "") {
      const newTicker = ticker.toUpperCase();
      setSelectedTicker(newTicker);
      fetchStockData(newTicker); // ✅ Fetch new data on search
    }
  };

  return (
    <div className="stocks-page" style={{ padding: "20px" }}>
      <div className="stock-header">
        <h2 style={{ textAlign: "center" }}>{selectedTicker} Stock Analysis</h2>
        {livePrice !== null && (
          <p style={{ textAlign: "center", fontSize: "18px", fontWeight: "bold", color: "green" }}>
            Live Price: ${livePrice.toFixed(2)}
          </p>
        )}
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

      <AddTicker />
      <LiveStockUpdate ticker={selectedTicker} />

      <StockChartComponent
        id="stockchart"
        primaryXAxis={{ valueType: "DateTime", labelFormat: "MMM dd" }}
        primaryYAxis={{ labelFormat: "${value}" }}
        tooltip={{ enable: true }}
        crosshair={{ enable: true }}
        periods={periods}
        key={selectedTicker} // ✅ Forces re-render when ticker changes
      >
        <Inject services={[
          DateTime, Tooltip, RangeTooltip, Crosshair, LineSeries, CandleSeries,
          Legend, Export, RsiIndicator, MacdIndicator, ScatterSeries
        ]} />
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

        {/* ✅ Highlight Entry Point */}
        {entryPoint !== null && (
          <StockChartSeriesDirective
            dataSource={[{ x: chartData[0]?.x, y: entryPoint }]}
            xName="x"
            yName="y"
            type="Scatter"
            marker={{ visible: true, shape: "Triangle", fill: "green", size: 10 }}
            name="Entry Point"
          />
        )}

        {/* ✅ Highlight Exit Point */}
        {exitPoint !== null && (
          <StockChartSeriesDirective
            dataSource={[{ x: chartData[chartData.length - 1]?.x, y: exitPoint }]}
            xName="x"
            yName="y"
            type="Scatter"
            marker={{ visible: true, shape: "InvertedTriangle", fill: "red", size: 10 }}
            name="Exit Point"
          />
        )}

      </StockChartComponent>
    </div>
  );
};

export default StocksPage;
