import React, { useState } from "react";
import DropdownMenu from "./DropdownMenu";
import DocsView from "./DocsView";
import Map from "./Map";
import TreemapView from "./TreemapView";
import QueryView from "./QueryView";
import Bourdieu from "./Bourdieu";
import { TopicsProvider } from "./UploadFileContext";

function App() {
  const [selectedView, setSelectedView] = useState("map"); // Default to 'map'

  return (
    <div className="App">
      <div className="main-display">
        <div className="top-right" id="top-banner">
          <a
            href="https://www.linkedin.com/company/bunka-ai/"
            target="_blank"
            rel="noopener noreferrer"
            className="linkedin-icon"
          >
            <img src="/linkedin_logo.png" alt="LinkedIn" />
          </a>
          <img src="/bunka_logo.png" alt="Bunka Logo" className="bunka-logo" />
          <DropdownMenu onSelectView={setSelectedView} />
        </div>
        <TopicsProvider onSelectView={setSelectedView}>
          {selectedView === "map" ? (
            <Map />
          ) : selectedView === "docs" ? (
            <DocsView />
          ) : selectedView === "treemap" ? (
            <TreemapView />
          ) : selectedView === "query" ? (
            <QueryView/>
          ) : selectedView === "bourdieu" ? (
            <Bourdieu />
          ) : (
            <QueryView />
          )}
        </TopicsProvider>
      </div>
    </div>
  );
}

export default App;
