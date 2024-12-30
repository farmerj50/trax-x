import React, { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";
import throttle from "lodash/throttle";


const CandlestickChart = ({ ticker, entryPoint, exitPoint, additionalData }) => {
  const chartContainerRef = useRef(null);
  const priceLineRef = useRef({});
  const candlestickSeriesRef = useRef(null);
  const chartRef = useRef(null);
  const isMounted = useRef(true);

  useEffect(() => {
    isMounted.current = true;
  
    // Initialize the chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth || 700,
      height: 400,
      layout: { backgroundColor: "#ffffff", textColor: "#000000" },
      grid: { vertLines: { color: "#eeeeee" }, horzLines: { color: "#eeeeee" } },
      priceScale: { borderColor: "#cccccc" },
      timeScale: { borderColor: "#cccccc" },
    });
  
    const candlestickSeries = chart.addCandlestickSeries();
    candlestickSeriesRef.current = candlestickSeries;
    chartRef.current = chart;
  
    const fetchInitialData = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/candlestick?ticker=${ticker}`);
        const data = await response.json();
  
        if (data.dates && data.open && data.high && data.low && data.close) {
          const formattedData = data.dates.map((date, index) => ({
            time: date,
            open: data.open[index],
            high: data.high[index],
            low: data.low[index],
            close: data.close[index],
          }));
          candlestickSeries.setData(formattedData);
  
          const currentPrice = formattedData[formattedData.length - 1].close;
          addOrUpdatePriceLine("currentPriceLine", currentPrice, "blue", `Current: ${currentPrice.toFixed(2)}`);
          addOrUpdatePriceLine("entryPriceLine", entryPoint, "green", `Entry: ${entryPoint.toFixed(2)}`);
          addOrUpdatePriceLine("exitPriceLine", exitPoint, "red", `Exit: ${exitPoint.toFixed(2)}`);
        }
      } catch (error) {
        if (isMounted.current) console.error("Error fetching initial data:", error);
      }
    };
  
    const addOrUpdatePriceLine = (key, price, color, title) => {
      if (priceLineRef.current[key]) {
        priceLineRef.current[key].applyOptions({ price, color, title });
      } else if (candlestickSeriesRef.current) {
        priceLineRef.current[key] = candlestickSeriesRef.current.createPriceLine({
          price,
          color,
          lineWidth: 2,
          title,
        });
      }
    };
  
    fetchInitialData();
  
    const throttledUpdateChart = throttle(async () => {
      if (!isMounted.current || !candlestickSeriesRef.current) return;
  
      try {
        const response = await fetch(`http://localhost:5000/api/candlestick?ticker=${ticker}`);
        const data = await response.json();
  
        if (data.dates && data.open && data.high && data.low && data.close) {
          const latestData = {
            time: data.dates[data.dates.length - 1],
            open: data.open[data.open.length - 1],
            high: data.high[data.high.length - 1],
            low: data.low[data.low.length - 1],
            close: data.close[data.close.length - 1],
          };
  
          candlestickSeriesRef.current.update(latestData);
          const currentPrice = latestData.close;
          addOrUpdatePriceLine("currentPriceLine", currentPrice, "blue", `Current: ${currentPrice.toFixed(2)}`);
        }
      } catch (error) {
        if (isMounted.current) console.error("Error updating chart:", error);
      }
    }, 5000);
  
    const intervalId = setInterval(() => {
      if (isMounted.current) throttledUpdateChart();
    }, 5000);
  
    const handleResize = () => {
      if (chartRef.current) chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
    };
  
    window.addEventListener("resize", handleResize);
  
    return () => {
      isMounted.current = false;
      clearInterval(intervalId);
      window.removeEventListener("resize", handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      candlestickSeriesRef.current = null;
      priceLineRef.current = {};
    };
  }, [ticker, entryPoint, exitPoint]);
  

  return (
    <div className="chart-container">
      <div className="chart-wrapper" ref={chartContainerRef}></div>
      <div className="chart-info">
        <h3 className="chart-title">{ticker}</h3>
        <div className="chart-stats">
          <p><strong>Volatility:</strong> {additionalData?.volatility || "N/A"}</p>
          <p><strong>Change:</strong> {additionalData?.change || "N/A"}</p>
          <p><strong>RSI:</strong> {additionalData?.rsi || "N/A"}</p>
        </div>
        <div className="price-info">
          <p><strong>Entry Point:</strong> ${entryPoint.toFixed(2)}</p>
          <p><strong>Exit Point:</strong> ${exitPoint.toFixed(2)}</p>
          <p><strong>Current Price:</strong> Displayed on chart</p>
        </div>
      </div>
    </div>
  );
};

export default CandlestickChart;
