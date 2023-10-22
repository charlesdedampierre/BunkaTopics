import React, { useEffect, useState } from 'react';

const DocsView = () => {
    const [docs, setDocs] = useState(null);

    useEffect(() => {
        // Fetch the content of "docs.json" when the component mounts
        fetch('/bunka_docs.json')
            .then((response) => response.json())
            .then((data) => {
                setDocs(data);
            })
            .catch((error) => {
                console.error('Error fetching JSON data:', error);
            });
    }, []);

    const downloadCSV = () => {
        // Create a CSV content string from the data
        const csvContent = "data:text/csv;charset=utf-8," + [
            ['Topic ID', 'Content'], // CSV header
            ...docs.map((doc) => [doc.topic_id, doc.content]), // CSV data
        ]
            .map((row) => row.map((cell) => `"${cell}"`).join(',')) // Wrap cells in double quotes
            .join('\n'); // Join rows with newline

        // Create a Blob containing the CSV data
        const blob = new Blob([csvContent], { type: 'text/csv' });

        // Create a download URL for the Blob
        const url = URL.createObjectURL(blob);

        // Create a temporary anchor element to trigger the download
        const a = document.createElement('a');
        a.href = url;
        a.download = 'docs.csv'; // Set the filename for the downloaded file
        a.click();

        // Revoke the URL to free up resources
        URL.revokeObjectURL(url);
    };

    return (
        <div className="docs-view">
            <h2>Documents View</h2>
            {docs ? (
                <div>
                    <button onClick={downloadCSV}>Download CSV</button>
                    <table>
                        <thead>
                            <tr>
                                <th>Topic ID</th>
                                <th>Content</th>
                            </tr>
                        </thead>
                        <tbody>
                            {docs.map((doc, index) => (
                                <tr key={index}>
                                    <td>{doc.topic_id}</td>
                                    <td>{doc.content}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <p>Loading...</p>
            )}
        </div>
    );
};

export default DocsView;
