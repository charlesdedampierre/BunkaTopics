import React, { useState } from 'react';
import DropdownMenu from './DropdownMenu';
import Map from './Map'; // Rename from JsonDisplay to Map
import DocsView from './DocsView';

const App = () => {
  const [selectedView, setSelectedView] = useState('map'); // Default to 'map'

  return (
    <div className="App">
      <div className="json-display">
        <div className="top-right">
          <a
            href="https://www.linkedin.com/company/company-profile-link"
            target="_blank"
            rel="noopener noreferrer"
            className="linkedin-icon"
          >
            <img src="/linkedin_logo.png" alt="LinkedIn" />
          </a>
          <img src="/bunka_logo.png" alt="Bunka Logo" className="bunka-logo" />
          <DropdownMenu onSelectView={setSelectedView} />
        </div>
        {selectedView === 'map' ? <Map /> : <DocsView />}
      </div>
    </div>
  );
};

export default App;
