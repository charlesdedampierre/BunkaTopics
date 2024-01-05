import React, { useState } from "react";
import Bourdieu from "./Bourdieu";
import DocsView from "./DocsView";
import DropdownMenu from "./DropdownMenu";
import MapView from "./Map";
import TreemapView from "./TreemapView";
import { TopicsProvider } from "./UploadFileContext";

function App() {
  const [selectedView, setSelectedView] = useState("map"); // Default to 'map'

  return (
    <div className="App">
      <div className="main-display">
        <div className="top-right" id="top-banner">
          <a href="https://www.linkedin.com/company/bunka-ai/" target="_blank" rel="noopener noreferrer" className="linkedin-icon">
            <img src="/linkedin_logo.png" alt="LinkedIn" />
          </a>
          <img src="/bunka_logo.png" alt="Bunka Logo" className="bunka-logo" />
          <DropdownMenu onSelectView={setSelectedView} selectedView={selectedView} />
        </div>
        <TopicsProvider onSelectView={setSelectedView} selectedView={selectedView}>
          {selectedView === "map" ? (
            <MapView />
          ) : selectedView === "docs" ? (
            <DocsView />
          ) : selectedView === "treemap" ? (
            /**
             * Hidden view for the moment
             */
            <TreemapView />
          ) : selectedView === "bourdieu" ? (
            <Bourdieu />
          ) : (
            <MapView />
          )}
        </TopicsProvider>
      </div>
    </div>
  );
}

export default App;
