import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import * as d3Contour from 'd3-contour';
import ReactDOM from 'react-dom';
import TextContainer from './TextContainer';


const Bourdieu = () => {
    const [jsonData, setJsonData] = useState(null);
    const [selectedDocument, setSelectedDocument] = useState(null);
    const svgRef = useRef(null);
    const textContainerRef = useRef(null);
    const scatterPlotContainerRef = useRef(null);
    const [queryData, setQueryData] = useState(null);

    const containerWidth = 2000; // Set the desired container width
    const containerHeight = 1500; // Set the desired container height

    const svgWidth = 1800; // Set the SVG width to match your map's desired width
    const svgHeight = 1500; // Set the SVG height to match your map's desired height

    useEffect(() => {
        const fetchData = async () => {
            try {
                const docsResponse = await fetch('/bunka_bourdieu_docs.json');
                const docsData = await docsResponse.json();
                setJsonData(docsData)

                const topicsResponse = await fetch('/bunka_bourdieu_topics.json');
                const topicsData = await topicsResponse.json();
                setJsonData(topicsData)

                // Fetch bunka_bourdieu_query.json
                const queryResponse = await fetch('/bunka_bourdieu_query.json');
                const queryData = await queryResponse.json();
                setQueryData(queryData);

                // You now have the data from bunka_bourdieu_query.json in queryData
                createScatterPlot(docsData, topicsData, queryData);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        fetchData();
    }, []);

    const createScatterPlot = (docsData, topicsData, queryData) => {
        const margin = { top: 20, right: 20, bottom: 50, left: 50 };
        const plotWidth = 1500;
        const plotHeight = 1100;


        const svg = d3.select(svgRef.current)
            .attr('width', svgWidth)
            .attr('height', svgHeight)
            .append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`)
            .style('background-color', 'blue');


        const xMin = d3.min(docsData, (d) => d.x);
        const xMax = d3.max(docsData, (d) => d.x);
        const yMin = d3.min(docsData, (d) => d.y);
        const yMax = d3.max(docsData, (d) => d.y);


        const maxDomainValue = Math.max(xMax, -xMin, yMax, -yMin);


        // Update the x and y scales to use the maximum value
        const xScale = d3.scaleLinear()
            .domain([-maxDomainValue, maxDomainValue]) // Updated domain for x scale
            .range([0, plotWidth]);

        const yScale = d3.scaleLinear()
            .domain([-maxDomainValue, maxDomainValue]) // Updated domain for y scale
            .range([plotHeight, 0]);

        // Replace text with BourdieuQuery words
        svg.append('text')
            .attr('x', xScale(xMin))
            .attr('y', yScale(0))
            .text(queryData.x_left_words[0])
            .style('text-anchor', 'start')
            .style('fill', 'purple');

        svg.append('text')
            .attr('x', xScale(xMax))
            .attr('y', yScale(0))
            .text(queryData.x_right_words[0])
            .style('text-anchor', 'end')
            .style('fill', 'purple');

        svg.append('text')
            .attr('x', xScale(0))
            .attr('y', yScale(yMax))
            .text(queryData.y_top_words[0])
            .style('text-anchor', 'start')
            .style('fill', 'purple');

        svg.append('text')
            .attr('x', xScale(0))
            .attr('y', yScale(yMin))
            .text(queryData.y_bottom_words[0])
            .style('text-anchor', 'middle')
            .style('fill', 'purple');



        const scatter = svg.selectAll('.scatter-point')
            .data(docsData)
            .enter()
            .append('circle')
            .attr('class', 'scatter-point')
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', 6) // Adjust the radius of the scatter points
            .style('fill', 'red') // Change the fill color
            .style('stroke', 'lightblue')
            .style('stroke-width', 1)
            .style('opacity', 0.8); // Adjust the opacity

        scatter.on('mouseover', (event, d) => {
            const content = d.content;

            // Create the content box div
            const contentBox = document.createElement('div');
            contentBox.id = 'content-box';
            contentBox.style.position = 'absolute';
            contentBox.style.left = (event.pageX + 10) + 'px'; // Offset the box position
            contentBox.style.top = (event.pageY - 20) + 'px'; // Offset the box position
            contentBox.style.backgroundColor = 'blue';
            contentBox.style.color = 'white';
            contentBox.style.padding = '10px';
            contentBox.innerHTML = content;

            // Append the content box to the body
            document.body.appendChild(contentBox);

            // Remove the content box on mouseout (hover out)
            scatter.on('mouseout', () => {
                document.body.removeChild(contentBox);
            });
        });


        const contourData = d3Contour.contourDensity()
            .x((d) => xScale(d.x))
            .y((d) => yScale(d.y))
            .size([plotWidth, plotHeight])
            .bandwidth(30)
            (docsData);

        const contourLineColor = 'rgb(94, 163, 252)';

        svg.selectAll('path.contour')
            .data(contourData)
            .enter()
            .append('path')
            .attr('class', 'contour')
            .attr('d', d3.geoPath())
            .style('fill', 'none')
            .style('stroke', contourLineColor)
            .style('stroke-width', 1);

        const topicsCentroids = topicsData.filter((d) => d.x_centroid && d.y_centroid);

        svg.selectAll('circle.topic-centroid')
            .data(topicsCentroids)
            .enter()
            .append('circle')
            .attr('class', 'topic-centroid')
            .attr('cx', (d) => xScale(d.x_centroid))
            .attr('cy', (d) => yScale(d.y_centroid))
            .attr('r', 8)
            .style('fill', 'red')
            .style('stroke', 'black')
            .style('stroke-width', 2)
            .on('click', (event, d) => {
                setSelectedDocument(d);
            });



        svg.selectAll('text.topic-label')
            .data(topicsCentroids)
            .enter()
            .append('text')
            .attr('class', 'topic-label')
            .attr('x', (d) => xScale(d.x_centroid))
            .attr('y', (d) => yScale(d.y_centroid) - 12)
            .text((d) => d.name)
            .style('text-anchor', 'middle');

        svg.append('line')
            .attr('x1', 0)
            .attr('x2', plotWidth)
            .attr('y1', yScale(0))
            .attr('y2', yScale(0))
            .style('stroke', 'black')
            .style('stroke-width', 3);

        // Add a thick line at x = 0
        svg.append('line')
            .attr('x1', xScale(0))
            .attr('x2', xScale(0))
            .attr('y1', 0)
            .attr('y2', plotHeight)
            .style('stroke', 'black')
            .style('stroke-width', 3);



        const convexHullData = topicsData.filter((d) => d.convex_hull);

        convexHullData.forEach((d) => {
            const hull = d.convex_hull;
            const hullPoints = hull.x_coordinates.map((x, i) => [xScale(x), yScale(hull.y_coordinates[i])]);

            svg.append('path')
                .datum(d3.polygonHull(hullPoints))
                .attr('class', 'convex-hull-polygon')
                .attr('d', (d) => "M" + d.join("L") + "Z")
                .style('fill', 'none')
                .style("stroke", "rgba(255, 255, 255, 0.5)")
                .style('stroke-width', 2);
        });

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
            .style('stroke-width', 2);

        let currentlyClickedPolygon = null;

        topicsPolygons.on('click', (event, d) => {
            if (currentlyClickedPolygon !== null) {
                currentlyClickedPolygon.style('stroke', 'transparent');
            }

            d3.select(event.target).style('stroke', 'red');
            currentlyClickedPolygon = d3.select(event.target);

            if (d.top_doc_content) {
                const topicName = d.name;
                const topicSize = d.size;
                const totalSize = topicsCentroids.reduce((sum, topic) => sum + topic.size, 0);
                const sizeFraction = Math.round((topicSize / totalSize) * 100);
                const content = d.top_doc_content;

                ReactDOM.render(
                    <TextContainer topicName={topicName} sizeFraction={sizeFraction} content={content} />,
                    textContainerRef.current
                );
            } else {
                textContainerRef.current.innerHTML = 'No content available for this topic.';
            }
        });




    };

    return (
        <div className="json-display">
            <div className="scatter-plot-and-text-container">
                <div className="scatter-plot-container" ref={scatterPlotContainerRef} style={{ width: containerWidth + 'px', height: containerHeight + 'px' }}>
                    <svg ref={svgRef}></svg>
                </div>
                <div className="text-container" ref={textContainerRef}>
                    {selectedDocument && (
                        <div className="text-content">
                            <h2 className="topic-name">Topic: {selectedDocument.topic_id}</h2>

                            <p>{selectedDocument.content}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Bourdieu;
