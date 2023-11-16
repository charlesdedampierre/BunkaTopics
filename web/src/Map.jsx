import React, {
  useEffect, useState, useRef, useContext,
} from "react";
import * as d3 from "d3";
import * as d3Contour from "d3-contour";
import ReactDOM from "react-dom"; // Import ReactDOM
import {
  Backdrop,
  CircularProgress,
} from "@mui/material";

import TextContainer from "./TextContainer";
import { TopicsContext } from "./UploadFileContext";

const bunkaDocs = "bunka_docs.json";
const bunkaTopics = "bunka_topics.json";
const { REACT_APP_API_ENDPOINT } = process.env;

function Map() {
  const [selectedDocument, setSelectedDocument] = useState(null);
  const { data: apiData, isLoading } = useContext(TopicsContext);

  const svgRef = useRef(null);
  const textContainerRef = useRef(null);
  const scatterPlotContainerRef = useRef(null);

  // const handleSearch = () => {
  //   const results = jsonData.filter((doc) =>
  //     doc.content.toLowerCase().includes(searchQuery.toLowerCase()),
  //   );
  //   setSearchResults(results);
  // };

  const createScatterPlot = (data) => {
    const margin = {
      top: 20,
      right: 20,
      bottom: 50,
      left: 50,
    };
    const plotWidth = window.innerWidth * 0.6;
    const plotHeight = window.innerHeight
        - document.getElementById("top-banner").clientHeight
        - 50; // Adjust the height as desired

    const svg = d3
      .select(svgRef.current)
      .attr("width", "100%")
      .attr("height", plotHeight)
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`)
      .style("background-color", "blue"); // Set the background color to blue

    const xMin = d3.min(data, (d) => d.x);
    const xMax = d3.max(data, (d) => d.x);
    const yMin = d3.min(data, (d) => d.y);
    const yMax = d3.max(data, (d) => d.y);

    const xScale = d3
      .scaleLinear()
      .domain([xMin, xMax]) // Use the full range of your data
      .range([0, plotWidth]);

    const yScale = d3
      .scaleLinear()
      .domain([yMin, yMax]) // Use the full range of your data
      .range([plotHeight, 0]);

    // Add contours
    const contourData = d3Contour
      .contourDensity()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .size([plotWidth, plotHeight])
      .bandwidth(30)(
        // Adjust the bandwidth as needed
        data,
      );

    // Define a custom color for the contour lines

    const contourLineColor = "rgb(94, 163, 252)";

    // Append the contour path to the SVG with a custom color
    svg
      .selectAll("path.contour")
      .data(contourData)
      .enter()
      .append("path")
      .attr("class", "contour")
      .attr("d", d3.geoPath())
      .style("fill", "none")
      .style("stroke", contourLineColor) // Set the contour line color to the custom color
      .style("stroke-width", 1);

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

    svg
      .selectAll("circle.topic-centroid")
      .data(topicsCentroids)
      .enter()
      .append("circle")
      .attr("class", "topic-centroid")
      .attr("cx", (d) => xScale(d.x_centroid))
      .attr("cy", (d) => yScale(d.y_centroid))
      .attr("r", 8) // Adjust the radius as needed
      .style("fill", "red") // Adjust the fill color as needed
      .style("stroke", "black")
      .style("stroke-width", 2)
      .on("click", (event, d) => {
        // Show the content and topic name of the clicked topic centroid in the text container
        setSelectedDocument(d);
      });

    // Add text labels for topic names
    svg
      .selectAll("text.topic-label")
      .data(topicsCentroids)
      .enter()
      .append("text")
      .attr("class", "topic-label")
      .attr("x", (d) => xScale(d.x_centroid))
      .attr("y", (d) => yScale(d.y_centroid) - 12) // Adjust the vertical position
      .text((d) => d.name) // Use the 'name' property for topic names
      .style("text-anchor", "middle"); // Center-align the text

    const convexHullData = data.filter((d) => d.convex_hull);

    convexHullData.forEach((d) => {
      const hull = d.convex_hull;
      const hullPoints = hull.x_coordinates.map((x, i) => [
        xScale(x),
        yScale(hull.y_coordinates[i]),
      ]);

      svg
        .append("path")
        .datum(d3.polygonHull(hullPoints))
        .attr("class", "convex-hull-polygon")
        .attr("d", (d1) => `M${d1.join("L")}Z`)
        .style("fill", "none")
        .style("stroke", "rgba(255, 255, 255, 0.5)") // White with 50% transparency
        .style("stroke-width", 2);
    });

    // Add polygons for topics. Delete if no clicking on polygons
    const topicsPolygons = svg
      .selectAll("polygon.topic-polygon")
      .data(topicsCentroids)
      .enter()
      .append("polygon")
      .attr("class", "topic-polygon")
      .attr("points", (d) => {
        const hull = d.convex_hull;
        const hullPoints = hull.x_coordinates.map((x, i) => [
          xScale(x),
          yScale(hull.y_coordinates[i]),
        ]);
        return hullPoints.map((point) => point.join(",")).join(" ");
      })
      .style("fill", "transparent")
      .style("stroke", "transparent")
      .style("stroke-width", 2); // Adjust the border width as needed

    let currentlyClickedPolygon = null;

    topicsPolygons.on("click", (event, d) => {
      // Reset the fill color of the previously clicked polygon to transparent light grey
      if (currentlyClickedPolygon !== null) {
        currentlyClickedPolygon.style("fill", "transparent");
        currentlyClickedPolygon.style("stroke", "transparent");
      }

      // Set the fill color of the clicked polygon to transparent light grey and add a red border
      const clickedPolygon = d3.select(event.target);
      clickedPolygon.style("fill", "rgba(200, 200, 200, 0.4)");
      clickedPolygon.style("stroke", "red");

      currentlyClickedPolygon = clickedPolygon;

      // Display the topic name and content from top_doc_content with a scroll system
      if (d.top_doc_content) {
        const topicName = d.name;
        const topicSize = d.size;
        const totalSize = topicsCentroids.reduce((sum, topic) => sum + topic.size, 0);
        const sizeFraction = Math.round((topicSize / totalSize) * 100);

        const content = d.top_doc_content;

        // Render the TextContainer component with topic details
        ReactDOM.render(
          <TextContainer
            topicName={topicName}
            sizeFraction={sizeFraction}
            content={content}
          />,
          textContainerRef.current,
        );
      } else {
        textContainerRef.current.innerHTML = "No content available for this topic.";
      }
    });
  };

  useEffect(() => {
    if (REACT_APP_API_ENDPOINT === "local" || apiData === undefined) {
      // Fetch the JSON data locally
      fetch(`/${bunkaDocs}`)
        .then((response) => response.json())
        .then((localData) => {
          // Fetch the local topics data and merge it with the existing data
          fetch(`/${bunkaTopics}`)
            .then((response) => response.json())
            .then((topicsData) => {
              // Merge the topics data with the existing data
              const mergedData = localData.concat(topicsData);

              // Call the function to create the scatter plot after data is loaded
              createScatterPlot(mergedData);
            })
            .catch((error) => {
              console.error("Error fetching topics data:", error);
            });
        })
        .catch((error) => {
          console.error("Error fetching JSON data:", error);
        });
    } else {
      // Call the function to create the scatter plot with the data provided by TopicsContext
      createScatterPlot(apiData);
    }
  }, [apiData]);

  return (
    <div className="json-display">
      {isLoading ? (
        <Backdrop open={isLoading} style={{ zIndex: 9999 }}>
          <CircularProgress color="primary" />
        </Backdrop>
      ) : (
        <div className="scatter-plot-and-text-container">
          <div className="scatter-plot-container" ref={scatterPlotContainerRef}>
            <svg ref={svgRef} />
          </div>
          <div className="text-container" ref={textContainerRef}>
            {selectedDocument && (
            <div className="text-content">
              <p>{selectedDocument.content}</p>
            </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default Map;
