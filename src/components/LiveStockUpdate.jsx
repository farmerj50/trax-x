import React, { useEffect, useState } from "react";
import { io } from "socket.io-client";

const socket = io("http://localhost:5000");

const LiveStockUpdates = ({ selectedTicker }) => {
  const [livePrice, setLivePrice] = useState(null);

  useEffect(() => {
    if (!selectedTicker) return;

    socket.on("stock_update", (data) => {
      if (data.ticker === selectedTicker) {
        console.log(`📡 Live Update Received for ${selectedTicker}:`, data.price);
        setLivePrice(data.price);
      }
    });

    return () => {
      socket.off("stock_update");
    };
  }, [selectedTicker]);

  return (
    <div>
      <h3>{selectedTicker} Live Price</h3>
      <p style={{ fontSize: "18px", fontWeight: "bold", color: "green" }}>
        {livePrice !== null ? `$${livePrice.toFixed(2)}` : "Waiting for updates..."}
      </p>
    </div>
  );
};

export default LiveStockUpdates;
