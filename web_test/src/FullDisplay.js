import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import * as d3Contour from 'd3-contour';

const JsonDisplay = () => {
    const [jsonData, setJsonData] = useState(null);
    const svgRef = useRef(null);
    const textContainerRef = useRef(null);

    useEffect(() => {
        // Fetch the JSON data
        fetch('/docs.json')
            .then((response) => response.json())
            .then((data) => {
                setJsonData(data);

                // Fetch the topics data and merge it with the existing data
                fetch('/topics.json')
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
        const plotWidth = 1000; // Adjust the width as desired
        const plotHeight = 1000; // Adjust the height as desired
        const fullWidth = plotWidth + margin.left + margin.right;
        const fullHeight = plotHeight + margin.top + margin.bottom;

        const svg = d3.select(svgRef.current)
            .attr('width', fullWidth)
            .attr('height', fullHeight)
            .append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`);

        const xScale = d3.scaleLinear()
            .domain([d3.min(data, (d) => d.x), d3.max(data, (d) => d.x)])
            .range([0, plotWidth]);

        const yScale = d3.scaleLinear()
            .domain([d3.min(data, (d) => d.y), d3.max(data, (d) => d.y)])
            .range([plotHeight, 0]);

        // Draw x and y axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);

        svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0, ${plotHeight})`)
            .call(xAxis);

        svg.append('g')
            .attr('class', 'y-axis')
            .call(yAxis);

        // Draw the zero lines
        svg.append('line')
            .attr('class', 'x-zero-line')
            .attr('x1', 0)
            .attr('x2', plotWidth)
            .attr('y1', yScale(0))
            .attr('y2', yScale(0))
            .style('stroke', 'black');

        svg.append('line')
            .attr('class', 'y-zero-line')
            .attr('x1', xScale(0))
            .attr('x2', xScale(0))
            .attr('y1', 0)
            .attr('y2', plotHeight)
            .style('stroke', 'black');

        // Add contours
        const contourData = d3Contour.contourDensity()
            .x((d) => xScale(d.x))
            .y((d) => yScale(d.y))
            .size([plotWidth, plotHeight])
            .bandwidth(30) // Adjust the bandwidth as needed
            (data);

        svg.selectAll('path.contour')
            .data(contourData)
            .enter()
            .append('path')
            .attr('class', 'contour')
            .attr('d', d3.geoPath())
            .style('fill', 'none')
            .style('stroke', 'black')
            .style('stroke-width', 1);



        const circles = svg.selectAll('circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', 5)
            .style('fill', 'steelblue')
            .on('click', (event, d) => {
                // Show the content of the clicked point in the text container
                const textContainer = d3.select(textContainerRef.current);
                textContainer.html(`
                    <p>${d.label}</p>
                    <p>${d.content}</p>
                `);
                // Change the color to pink on click
                circles.style('fill', (pointData) => (pointData === d) ? 'pink' : 'steelblue');
            });

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
                // Show the content of the clicked topic centroid in the text container
                const textContainer = d3.select(textContainerRef.current);
                textContainer.html(`
                    <p>${d.name}</p>
                `);
            });

        const convexHullData = data.filter((d) => d.convex_hull);

        convexHullData.forEach((d) => {
            const hull = d.convex_hull;
            const hullPoints = hull.x_coordinates.map((x, i) => [xScale(x), yScale(hull.y_coordinates[i])]);

            svg.append('path')
                .datum(d3.polygonHull(hullPoints))
                .attr('class', 'convex-hull-polygon')
                .attr('d', (d) => "M" + d.join("L") + "Z")
                .style('fill', 'none')
                .style('stroke', 'green') // Adjust the stroke color as needed
                .style('stroke-width', 2);
        });


    };

    return (
        <div className="json-display">
            <h2>Bunka</h2>
            <div className="scatter-plot-and-text-container">
                <div className="scatter-plot-container">
                    <svg ref={svgRef}></svg>
                </div>
                <div className="text-container" ref={textContainerRef}></div>
            </div>
        </div>
    );
};

export default JsonDisplay;

