import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import * as d3Contour from 'd3-contour';
import ReactDOM from 'react-dom';
import TextContainer from './TextContainer';


const Bourdieu = () => {
    const [jsonData, setJsonData] = useState(null);
    const [selectedDocument, setSelectedDocument] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const svgRef = useRef(null);
    const textContainerRef = useRef(null);
    const scatterPlotContainerRef = useRef(null);
    const [queryData, setQueryData] = useState(null);


    useEffect(() => {
        const fetchData = async () => {
            try {
                const docsResponse = await fetch('/bunka_bourdieu_docs.json');
                const docsData = await docsResponse.json();

                const topicsResponse = await fetch('/bunka_bourdieu_topics.json');
                const topicsData = await topicsResponse.json();

                const mergedData = docsData.concat(topicsData);
                setJsonData(mergedData);

                // Fetch bunka_bourdieu_query.json
                const queryResponse = await fetch('/bunka_bourdieu_query.json');
                const queryData = await queryResponse.json();
                setQueryData(queryData);

                // You now have the data from bunka_bourdieu_query.json in queryData

                createScatterPlot(mergedData, queryData);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        fetchData();
    }, []);


    const createScatterPlot = (data, queryData) => {
        const margin = { top: 20, right: 20, bottom: 50, left: 50 };
        const plotWidth = 1500;
        const plotHeight = 1100;
        const fullWidth = plotWidth + margin.left + margin.right;
        const fullHeight = plotHeight + margin.top + margin.bottom;



        const svg = d3.select(svgRef.current)
            .attr('width', fullWidth)
            .attr('height', fullHeight)
            .append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`)
            .style('background-color', 'blue');



        const xMin = d3.min(data, (d) => d.x);
        const xMax = d3.max(data, (d) => d.x);
        const yMin = d3.min(data, (d) => d.y);
        const yMax = d3.max(data, (d) => d.y);


        const maxDomainValue = Math.max(xMin, xMax, yMin, yMax);

        // Update the x and y scales to use the maximum value
        const xScale = d3.scaleLinear()
            .domain([-maxDomainValue, maxDomainValue]) // Updated domain for x scale
            .range([0, plotWidth]);

        const yScale = d3.scaleLinear()
            .domain([-maxDomainValue, maxDomainValue]) // Updated domain for y scale
            .range([plotHeight, 0]);

        const scatter = svg.selectAll('.scatter-point')
            .data(data)
            .enter()
            .append('circle')
            .attr('class', 'scatter-point')
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', 4) // Adjust the radius of the scatter points
            .style('fill', 'red') // Change the fill color
            .style('stroke', 'black')
            .style('stroke-width', 1)
            .style('opacity', 0.8); // Adjust the opacity

        // Add tooltips on hover
        scatter.on('mouseover', (event, d) => {
            const topicName = d.name;
            const topicSize = d.size;
            const totalSize = data.reduce((sum, topic) => sum + topic.size, 0);
            const sizeFraction = Math.round((topicSize / totalSize) * 100);
            const content = d.top_doc_content;

            // You can display the tooltip using a pop-up or a TextContainer component.
            // Here's an example of using a pop-up div for the tooltip.
            const tooltip = d3.select('.tooltip');
            tooltip.style('display', 'block');
            tooltip.html(`
                <strong>Topic:</strong> ${topicName}<br>
                <strong>Size Fraction:</strong> ${sizeFraction}%<br>
                <strong>Content:</strong> ${content}
            `);
        });

        scatter.on('mouseout', () => {
            // Hide the tooltip on mouseout
            d3.select('.tooltip').style('display', 'none');
        });


        const contourData = d3Contour.contourDensity()
            .x((d) => xScale(d.x))
            .y((d) => yScale(d.y))
            .size([plotWidth, plotHeight])
            .bandwidth(30)
            (data);

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

        const topicsCentroids = data.filter((d) => d.x_centroid && d.y_centroid);

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

        const convexHullData = data.filter((d) => d.convex_hull);

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

        const minX = d3.min(data, (d) => d.x);
        const maxX = d3.max(data, (d) => d.x);
        const minY = d3.min(data, (d) => d.y);
        const maxY = d3.max(data, (d) => d.y);

        // Replace text with BourdieuQuery words
        svg.append('text')
            .attr('x', xScale(minX))
            .attr('y', yScale(0))
            .text(queryData.x_left_words[0])
            .style('text-anchor', 'start')
            .style('fill', 'black');

        svg.append('text')
            .attr('x', xScale(maxX))
            .attr('y', yScale(0))
            .text(queryData.x_right_words[0])
            .style('text-anchor', 'end')
            .style('fill', 'black');

        svg.append('text')
            .attr('x', xScale(0))
            .attr('y', yScale(maxY))
            .text(queryData.y_top_words[0])
            .style('text-anchor', 'start')
            .style('fill', 'black');

        svg.append('text')
            .attr('x', xScale(0))
            .attr('y', yScale(minY))
            .text(queryData.y_bottom_words[0])
            .style('fill', 'black');


    };

    const handleSearch = (event) => {
        const query = event.target.value;
        setSearchQuery(query);
        const filteredData = jsonData.filter((doc) =>
            doc.content.toLowerCase().includes(query.toLowerCase())
        );
        setSelectedDocument(filteredData[0]);
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

export default Bourdieu;
