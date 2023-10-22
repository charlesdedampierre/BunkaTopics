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

    return (
        <div className="docs-view">
            <h2>Documents View</h2>
            {docs ? (
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
            ) : (
                <p>Loading...</p>
            )}
        </div>
    );
};

export default DocsView;
