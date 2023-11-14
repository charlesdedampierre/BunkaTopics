// GetData.js

import React from "react";

function GetData() {
  return (
    <div>
      <h1>Bunka Topics Analysis</h1>
      <form id="bunkaForm" encType="multipart/form-data">
        <input type="file" id="csvFile" name="csvFile" accept=".csv" required />
        <input
          type="text"
          id="columnName"
          name="columnName"
          placeholder="Enter column name"
          required
        />
        <button type="submit">Analyze</button>
      </form>

      <div id="result" />
    </div>
  );
}

export default GetData;
