import React, { useEffect, useState } from "react";

const TickerNewsWidget = ({ ticker }) => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTickerNews = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/ticker-news?ticker=${ticker}`);
        const data = await response.json();
        if (data.results) {
          setNews(data.results);
        } else {
          setNews([]);
        }
      } catch (error) {
        console.error("Error fetching news:", error);
        setNews([]);
      } finally {
        setLoading(false);
      }
    };

    if (ticker) {
      fetchTickerNews();
    }
  }, [ticker]);

  if (loading) {
    return <div>Loading news...</div>;
  }

  if (news.length === 0) {
    return <div>No news available for {ticker}.</div>;
  }

  return (
    <div className="ticker-news-widget" style={{ padding: "10px" }}>
      <h4 style={{ marginBottom: "10px" }}>Latest News for {ticker}</h4>
      {news.map((article, index) => (
        <div key={index} style={{ marginBottom: "15px" }}>
          <a
            href={article.article_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: "16px", fontWeight: "bold", textDecoration: "none" }}
          >
            {article.title}
          </a>
          <p style={{ fontSize: "14px", color: "#555" }}>{article.description}</p>
        </div>
      ))}
    </div>
  );
};

export default TickerNewsWidget;
