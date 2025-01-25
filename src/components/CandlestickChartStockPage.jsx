import React, { useEffect, useState } from "react";
import {
  ChartComponent,
  SeriesCollectionDirective,
  SeriesDirective,
  Inject,
  DateTime,
  CandleSeries,
  LineSeries,
  Tooltip,
  Zoom,
  Crosshair,
} from "@syncfusion/ej2-react-charts";
import { registerLicense } from "@syncfusion/ej2-base";

// Register Syncfusion license
registerLicense("Ngo9BigBOggjHTQxAR8/V1NMaF5cXmRCf1FpRmJGdld5fUVHYVZUTXxaS00DNHVRdkdmWX5ednVWQ2BfVEJ+WEY=");

const CandlestickChartStockPage = ({ ticker }) => {
  const [chartData, setChartData] = useState([]);
  const [lineSeriesData, setLineSeriesData] = useState([]); // Data for LineSeries
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchChartData = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await fetch(
          `http://localhost:5000/api/candlestick?ticker=${ticker}`
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

          // Optionally create a separate dataset for LineSeries
          const lineData = data.dates.map((date, index) => ({
            x: new Date(date),
            y: data.close[index], // Using "close" values for LineSeries
          }));

          setChartData(formattedData);
          setLineSeriesData(lineData);
        } else {
          setError("No data available for this stock.");
        }
      } catch (err) {
        setError("Failed to load chart data.");
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  }, [ticker]);

  if (loading) {
    return <p style={{ textAlign: "center" }}>Loading chart...</p>;
  }

  if (error) {
    return <p style={{ color: "red", textAlign: "center" }}>{error}</p>;
  }

  if (chartData.length === 0) {
    return <p style={{ textAlign: "center" }}>No data available for {ticker}.</p>;
  }

  return (
    <div className="chart-container-stock-page" style={{ height: "100%", width: "100%" }}>
      <ChartComponent
        id={`chart-${ticker}`}
        primaryXAxis={{
          valueType: "DateTime",
          labelFormat: "MMM dd",
          intervalType: "Days",
          crosshairTooltip: { enable: true },
        }}
        primaryYAxis={{
          labelFormat: "${value}",
          crosshairTooltip: { enable: true },
        }}
        tooltip={{ enable: true }}
        crosshair={{
          enable: true,
          lineType: "Both",
        }}
        zoomSettings={{
          enableMouseWheelZooming: true,
          enablePinchZooming: true,
          mode: "XY",
        }}
        height="100%" // Ensures chart adjusts to parent container
        width="100%" // Ensures chart adjusts to parent container
      >
        <Inject
          services={[CandleSeries, LineSeries, DateTime, Tooltip, Zoom, Crosshair]}
        />
        <SeriesCollectionDirective>
          {/* CandleSeries for candlestick data */}
          <SeriesDirective
            dataSource={chartData}
            xName="x"
            open="open"
            high="high"
            low="low"
            close="close"
            type="Candle"
          />
          {/* LineSeries for additional line data */}
          <SeriesDirective
            dataSource={lineSeriesData}
            xName="x"
            yName="y"
            type="Line"
            width={2}
            name="Close Prices" // Legend name for LineSeries
          />
        </SeriesCollectionDirective>
      </ChartComponent>
    </div>
  );
};

export default CandlestickChartStockPage;
