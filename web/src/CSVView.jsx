import React, { useState } from "react";

function CSVView({ onCSVImport }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleImport = () => {
    if (selectedFile) {
      // You can implement the logic to handle the selected CSV file here
      // For example, you can read and process the CSV data.
      // Once you've processed the data, you can call the `onCSVImport` callback
      // to pass the data back to the parent component.
    }
  };

  return (
    <div>
      <h2>Import CSV</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleImport}>Import CSV</button>
    </div>
  );
}

export default CSVView;
