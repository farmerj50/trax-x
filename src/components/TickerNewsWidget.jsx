import React, { useEffect, useState } from "react";

const TickerNewsWidget = ({ tickers }) => {
  const [news, setNews] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    console.log("📌 Received tickers in TickerNewsWidget:", tickers);

    if (!tickers || tickers.length === 0) {
      console.warn("⚠️ No tickers available for fetching news.");
      setNews({});
      setLoading(false);
      return;
    }

    const fetchNewsForTickers = async () => {
      setLoading(true);
      const tickerString = tickers.join(",");  // ✅ Convert to comma-separated string
      console.log(`📌 Fetching news for tickers: ${tickerString}`);

      try {
        const response = await fetch(`http://localhost:5000/api/ticker-news?ticker=${tickerString}`);
        const data = await response.json();
        console.log("📌 API Response:", data);

        if (data.error) {
          console.error("❌ Error from API:", data.error);
          setNews({});
        } else {
          setNews(data);
        }
      } catch (error) {
        console.error("❌ Error fetching news:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchNewsForTickers();
  }, [tickers]);

  if (loading) {
    return <div className="news-widget">Loading news...</div>;
  }

  if (Object.keys(news).length === 0) {
    return <div className="news-widget">No news available for the selected stocks.</div>;
  }

  return (
    <div className="news-widget">
      <h4>Latest Stock News</h4>
      {Object.entries(news).map(([ticker, articles]) => (
        <div key={ticker} className="news-section">
          <h5>{ticker}</h5>
          {articles.length > 0 ? (
            articles.map((article) => (
              <div key={article.id} className="news-article">
                <h6>
                  <a href={article.article_url} target="_blank" rel="noopener noreferrer">
                    {article.title}
                  </a>
                </h6>
                <p><strong>Author:</strong> {article.author}</p>

                {/* ✅ FIX: Extract publisher name properly */}
                <p><strong>Publisher:</strong> {typeof article.publisher === "object" ? article.publisher.name : "Unknown Publisher"}</p>

                <p><strong>Published:</strong> {new Date(article.published_utc).toLocaleString()}</p>
                {article.description && <p>{article.description}</p>}
                {article.image_url && <img src={article.image_url} alt={article.title} style={{ maxWidth: "100%" }} />}
                <p>
                  <strong>Sentiment:</strong> {article.sentiment} - {article.sentiment_reasoning}
                </p>
              </div>
            ))
          ) : (
            <p>No news available for {ticker}.</p>
          )}
        </div>
      ))}
    </div>
  );
};

export default TickerNewsWidget;
