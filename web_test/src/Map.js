import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import * as d3Contour from 'd3-contour';

const Map = () => {
    const [jsonData, setJsonData] = useState(null);
    const [selectedDocument, setSelectedDocument] = useState(null);
    const [searchQuery, setSearchQuery] = useState(''); // State for search input
    const svgRef = useRef(null);
    const textContainerRef = useRef(null);
    const scatterPlotContainerRef = useRef(null);

    useEffect(() => {
        // Fetch the JSON data
        fetch('/bunka_docs.json')
            .then((response) => response.json())
            .then((data) => {
                setJsonData(data);

                // Fetch the topics data and merge it with the existing data
                fetch('/bunka_topics.json')
                    .then((response) => response.json())
                    .then((topicsData) => {
                        // Merge the topics data with the existing data
                        const mergedData = data.concat(topicsData);

                        // Call the function to create the scatter plot after data is loaded
                        createScatterPlot(mergedData);
                    })
                    .catch((error) => {
                        console.error('Error fetching topics data:', error);
                    });
            })
            .catch((error) => {
                console.error('Error fetching JSON data:', error);
            });
    }, []);

    const createScatterPlot = (data) => {
        const margin = { top: 20, right: 20, bottom: 50, left: 50 };
        const plotWidth = 1500; // Adjust the width as desired
        const plotHeight = 1100; // Adjust the height as desired
        const fullWidth = plotWidth + margin.left + margin.right;
        const fullHeight = plotHeight + margin.top + margin.bottom;

        const svg = d3.select(svgRef.current)
            .attr('width', fullWidth)
            .attr('height', fullHeight)
            .append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`)
            .style('background-color', 'blue'); // Set the background color to blue


        const xMin = d3.min(data, (d) => d.x);
        const xMax = d3.max(data, (d) => d.x);
        const yMin = d3.min(data, (d) => d.y);
        const yMax = d3.max(data, (d) => d.y);

        const xScale = d3.scaleLinear()
            .domain([xMin, xMax]) // Use the full range of your data
            .range([0, plotWidth]);

        const yScale = d3.scaleLinear()
            .domain([yMin, yMax]) // Use the full range of your data
            .range([plotHeight, 0]);

        // Add contours
        const contourData = d3Contour.contourDensity()
            .x((d) => xScale(d.x))
            .y((d) => yScale(d.y))
            .size([plotWidth, plotHeight])
            .bandwidth(30) // Adjust the bandwidth as needed
            (data);

        // Define a custom color for the contour lines


        const contourLineColor = 'rgb(94, 163, 252)';

        // Append the contour path to the SVG with a custom color
        svg.selectAll('path.contour')
            .data(contourData)
            .enter()
            .append('path')
            .attr('class', 'contour')
            .attr('d', d3.geoPath())
            .style('fill', 'none')
            .style('stroke', contourLineColor) // Set the contour line color to the custom color
            .style('stroke-width', 1);



        /*
        const circles = svg.selectAll('circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', 5)
            .style('fill', 'lightblue')
            .on('click', (event, d) => {
                // Show the content and topic name of the clicked point in the text container
                setSelectedDocument(d);
                // Change the color to pink on click
                circles.style('fill', (pointData) => (pointData === d) ? 'pink' : 'lightblue');
            });
            */

        const topicsCentroids = data.filter((d) => d.x_centroid && d.y_centroid);

        svg.selectAll('circle.topic-centroid')
            .data(topicsCentroids)
            .enter()
            .append('circle')
            .attr('class', 'topic-centroid')
            .attr('cx', (d) => xScale(d.x_centroid))
            .attr('cy', (d) => yScale(d.y_centroid))
            .attr('r', 8) // Adjust the radius as needed
            .style('fill', 'red') // Adjust the fill color as needed
            .style('stroke', 'black')
            .style('stroke-width', 2)
            .on('click', (event, d) => {
                // Show the content and topic name of the clicked topic centroid in the text container
                setSelectedDocument(d);
            });

        // Add text labels for topic names
        svg.selectAll('text.topic-label')
            .data(topicsCentroids)
            .enter()
            .append('text')
            .attr('class', 'topic-label')
            .attr('x', (d) => xScale(d.x_centroid))
            .attr('y', (d) => yScale(d.y_centroid) - 12) // Adjust the vertical position
            .text((d) => d.name) // Use the 'name' property for topic names
            .style('text-anchor', 'middle'); // Center-align the text


        const convexHullData = data.filter((d) => d.convex_hull);

        convexHullData.forEach((d) => {
            const hull = d.convex_hull;
            const hullPoints = hull.x_coordinates.map((x, i) => [xScale(x), yScale(hull.y_coordinates[i])]);

            svg.append('path')
                .datum(d3.polygonHull(hullPoints))
                .attr('class', 'convex-hull-polygon')
                .attr('d', (d) => "M" + d.join("L") + "Z")
                .style('fill', 'none')
                .style("stroke", "rgba(255, 255, 255, 0.5)") // White with 50% transparency
                .style('stroke-width', 2);
        });

        // Add polygons for topics. Delete if no clicking on polygons
        const topicsPolygons = svg
            .selectAll('polygon.topic-polygon')
            .data(topicsCentroids)
            .enter()
            .append('polygon')
            .attr('class', 'topic-polygon')
            .attr('points', (d) => {
                const hull = d.convex_hull;
                const hullPoints = hull.x_coordinates.map((x, i) => [xScale(x), yScale(hull.y_coordinates[i])]);
                return hullPoints.map((point) => point.join(',')).join(' ');
            })
            .style('fill', 'transparent')
            .style('stroke', 'transparent')
            .style('stroke-width', 2); // Adjust the border width as needed

        let currentlyClickedPolygon = null;

        topicsPolygons.on('click', (event, d) => {
            if (currentlyClickedPolygon !== null) {
                // Reset the previously clicked polygon's border to transparent
                currentlyClickedPolygon.style('stroke', 'transparent');
            }

            // Set the color of the clicked polygon's border to red
            d3.select(event.target).style('stroke', 'red');

            currentlyClickedPolygon = d3.select(event.target);

            // Display the topic name and content from top_doc_content with a scroll system
            if (d.top_doc_content) {
                const topicName = d.name;
                const topicSize = d.size;
                const totalSize = topicsCentroids.reduce((sum, topic) => sum + topic.size, 0);
                const sizeFraction = Math.round((topicSize / totalSize) * 100);
                const content = d.top_doc_content.map((doc, index) => (
                    `<div class="box" key=${index}>
                ${doc}
            </div>`
                )).join('');

                // Set a max height and overflow for the text container
                textContainerRef.current.style.maxHeight = '1100px'; // Adjust the height as needed
                textContainerRef.current.style.maxWitdh = '600'; // Adjust the height as needed

                textContainerRef.current.style.overflow = 'auto';

                // Display the topic name on top, followed by the content
                textContainerRef.current.innerHTML = `
            <div class="topic-box">
                <h2>${topicName}</h2>
                <h3>${sizeFraction}% of the Territory</h3>
                <div class="documents-list">
                    ${content}
                </div>
            </div>
        `;

                // Add click event listeners to each box element
                const boxes = textContainerRef.current.querySelectorAll('.box');
                boxes.forEach(box => {
                    box.addEventListener('click', () => {
                        // Toggle the "clicked" class to change the background color
                        box.classList.toggle('clicked');
                    });
                });
            } else {
                textContainerRef.current.innerHTML = 'No content available for this topic.';
            }
        });

        /*
        // Add a button to take a screenshot
        const screenshotButton = document.createElement('button');
        screenshotButton.innerText = 'Take Screenshot';
        screenshotButton.style.position = 'absolute';
        screenshotButton.style.bottom = '10px'; // Position at the bottom of the scatter plot container
        screenshotButton.style.left = '50%'; // Center horizontally
        screenshotButton.style.transform = 'translateX(-50%)'; // Center horizontally
        screenshotButton.style.padding = '10px 20px'; // Increase padding for a larger button
        screenshotButton.style.background = 'darkblue'; // Set the background color to blue
        screenshotButton.style.color = 'white'; // Set text color to white
        screenshotButton.style.border = 'darkblue'; // Add a dark blue border
        screenshotButton.style.fontSize = '16px'; // Increase font size
   


        screenshotButton.addEventListener('click', () => {
            // Use html2canvas to capture a screenshot of the scatter plot container
            html2canvas(scatterPlotContainerRef.current).then((canvas) => {
                const screenshot = canvas.toDataURL('image/png');

                // Create a temporary anchor element to trigger the download
                const a = document.createElement('a');
                a.href = screenshot;
                a.download = 'bunka_map.png'; // You can change the filename as desired
                a.click();
            });
        });

        // Append the screenshot button to the scatter plot container
        scatterPlotContainerRef.current.appendChild(screenshotButton);

        */

    };


    const handleSearch = (event) => {
        const query = event.target.value;
        setSearchQuery(query);
        // You can implement your search logic here, e.g., filtering data based on the query
        // and updating the displayed content.
        // For simplicity, let's filter based on the "content" property.
        const filteredData = jsonData.filter((doc) =>
            doc.content.toLowerCase().includes(query.toLowerCase())
        );
        setSelectedDocument(filteredData[0]); // Set the first matching document as selected
    };

    return (
        <div className="json-display">
            <div className="scatter-plot-and-text-container">
                <div className="scatter-plot-container" ref={scatterPlotContainerRef}>
                    <svg ref={svgRef}></svg>
                </div>
                <div className="text-container" ref={textContainerRef}>
                    {selectedDocument && (
                        <div className="text-content">
                            <h2 className="topic-name">Topic: {selectedDocument.topic_id}</h2>
                            <input
                                type="text"
                                placeholder="Search..."
                                value={searchQuery}
                                onChange={handleSearch}
                            />
                            <p>{selectedDocument.content}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Map;






