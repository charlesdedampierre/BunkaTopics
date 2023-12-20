import * as d3 from "d3";
import { ZoomTransform } from "d3";
import * as d3Contour from "d3-contour";
import { Backdrop, CircularProgress, Box, Button } from "@mui/material";
import React, { useCallback, useEffect, useRef, useState, useContext } from "react";
import ReactDOM from "react-dom";
import TextContainer from "./TextContainer";
import { TopicsContext } from "./UploadFileContext";
import QueryView from "./QueryView";
import HelpIcon from '@mui/icons-material/Help';
import Tooltip from '@mui/material/Tooltip';

const bunkaDocs = "bunka_bourdieu_docs.json";
const bunkaTopics = "bunka_bourdieu_topics.json";
const bunkaQuery = "bunka_bourdieu_query.json";
const { REACT_APP_API_ENDPOINT } = process.env;

function Bourdieu() {
  const [selectedDocument, setSelectedDocument] = useState(null);
  const { bourdieuData: apiData, isLoading } = useContext(TopicsContext);

  const svgRef = useRef(null);
  const textContainerRef = useRef(null);
  const scatterPlotContainerRef = useRef(null);
  // Set the SVG height to match your map's desired height
  const svgHeight = window.innerHeight - document.getElementById("top-banner").clientHeight - 50;
  const svgWidth = window.innerWidth * 0.65; // Set the svg container height to match the layout

  const createScatterPlot = useCallback(
    (docsData, topicsData, queryData) => {
      const margin = {
        top: 20,
        right: 20,
        bottom: 50,
        left: 50,
      };
      const plotWidth = svgWidth;
      const plotHeight = svgHeight;

      const svg = d3
        .select(svgRef.current)
        .attr("width", "100%")
        .attr("height", svgHeight);
      /**
       * SVG canvas group on which transforms apply.
       */
      const g = svg.append("g")
      .classed("canvas", true)
      .attr("transform", `translate(${margin.left}, ${margin.top})`);
      /**
       * Zoom.
       */
      const zoom = d3.zoom()
        .scaleExtent([1, 3])
        .translateExtent([[0,0], [plotWidth, plotHeight]])
        .on("zoom", function ({ transform }) {
          g.attr(
            "transform",
            `translate(${transform.x ?? 0}, ${transform.y ?? 0}) scale(${transform.k ?? 1})`
          )
          //positionLabels()
          // props.setTransform?.({
          //   x: transform.x,
          //   y: transform.y,
          //   k: transform.k
          // })
        });
      svg.call(zoom);

      /**
       * Initial zoom.
       */
      const defaultTransform = { k: 1 };
      const initialTransform = defaultTransform?.k != null
        ? new ZoomTransform(
          defaultTransform.k ?? 1,
          defaultTransform.x ?? 0,
          defaultTransform.y ?? 0
        )
        : d3.zoomIdentity;
      svg.call(zoom.transform, initialTransform);

      // Axes
      const xMin = d3.min(docsData, (d) => d.x);
      const xMax = d3.max(docsData, (d) => d.x);
      const yMin = d3.min(docsData, (d) => d.y);
      const yMax = d3.max(docsData, (d) => d.y);

      const maxDomainValue = Math.max(xMax, -xMin, yMax, -yMin);

      // Update the x and y scales to use the maximum value
      const xScale = d3
        .scaleLinear()
        .domain([-maxDomainValue, maxDomainValue]) // Updated domain for x scale
        .range([0, plotWidth]);

      const yScale = d3
        .scaleLinear()
        .domain([-maxDomainValue, maxDomainValue]) // Updated domain for y scale
        .range([plotHeight, 0]);

      // Replace text with BourdieuQuery words
      svg
        .append("text")
        .attr("x", xScale(xMin))
        .attr("y", yScale(0.01))
        .text(queryData.x_right_words[0])
        .style("text-anchor", "start")
        .style("fill", "blue");

      svg
        .append("text")
        .attr("x", xScale(xMax))
        .attr("y", yScale(0.01))
        .text(queryData.x_left_words[0])
        .style("text-anchor", "start")
        .style("fill", "blue");

      svg
        .append("text")
        .attr("x", xScale(0.01))
        .attr("y", yScale(yMax))
        .text(queryData.y_top_words[0])
        .style("text-anchor", "start")
        .style("fill", "blue");

      svg
        .append("text")
        .attr("x", xScale(0.01))
        .attr("y", yScale(yMin))
        .text(queryData.y_bottom_words[0])
        .style("text-anchor", "end")
        .style("fill", "blue");

      /*
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
            */

      const contourData = d3Contour
        .contourDensity()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y))
        .size([plotWidth, plotHeight])
        .bandwidth(30)(docsData);

      const contourLineColor = "rgb(94, 163, 252)";

      svg
        .selectAll("path.contour")
        .data(contourData)
        .enter()
        .append("path")
        .attr("class", "contour")
        .attr("d", d3.geoPath())
        .style("fill", "none")
        .style("stroke", contourLineColor)
        .style("stroke-width", 1);

      const topicsCentroids = topicsData.filter((d) => d.x_centroid && d.y_centroid);

      svg
        .selectAll("circle.topic-centroid")
        .data(topicsCentroids)
        .enter()
        .append("circle")
        .attr("class", "topic-centroid")
        .attr("cx", (d) => xScale(d.x_centroid))
        .attr("cy", (d) => yScale(d.y_centroid))
        .attr("r", 8)
        .style("fill", "red")
        .style("stroke", "black")
        .style("stroke-width", 2)
        .on("click", (event, d) => {
          setSelectedDocument(d);
        });

      svg
        .selectAll("text.topic-label")
        .data(topicsCentroids)
        .enter()
        .append("text")
        .attr("class", "topic-label")
        .attr("x", (d) => xScale(d.x_centroid))
        .attr("y", (d) => yScale(d.y_centroid) - 12)
        .text((d) => d.name)
        .style("text-anchor", "middle");

      svg.append("line").attr("x1", 0).attr("x2", plotWidth).attr("y1", yScale(0)).attr("y2", yScale(0)).style("stroke", "black").style("stroke-width", 3);

      // Add a thick line at x = 0
      svg.append("line").attr("x1", xScale(0)).attr("x2", xScale(0)).attr("y1", 0).attr("y2", plotHeight).style("stroke", "black").style("stroke-width", 3);

      const convexHullData = topicsData.filter((d) => d.convex_hull);
      for (const d of convexHullData) {
        const hull = d.convex_hull;
        if (hull) {
          const hullPoints = hull.x_coordinates.map((x, i) => [xScale(x), yScale(hull.y_coordinates[i])]);

          svg
            .append("path")
            .datum(d3.polygonHull(hullPoints))
            .attr("class", "convex-hull-polygon")
            .attr("d", (dAttr) => `M${dAttr.join("L")}Z`)
            .style("fill", "none")
            .style("stroke", "rgba(255, 255, 255, 0.5)")
            .style("stroke-width", 2);
        }
      }
      const xGreaterThanZeroAndYGreaterThanZero = docsData.filter((d) => d.x > 0 && d.y > 0).length;
      const xLessThanZeroAndYGreaterThanZero = docsData.filter((d) => d.x < 0 && d.y > 0).length;
      const xGreaterThanZeroAndYLessThanZero = docsData.filter((d) => d.x > 0 && d.y < 0).length;
      const xLessThanZeroAndYLessThanZero = docsData.filter((d) => d.x < 0 && d.y < 0).length;

      // Calculate the total number of documents
      const totalDocuments = docsData.length;

      // Calculate the percentages
      const percentageXGreaterThanZeroAndYGreaterThanZero = (xGreaterThanZeroAndYGreaterThanZero / totalDocuments) * 100;
      const percentageXLessThanZeroAndYGreaterThanZero = (xLessThanZeroAndYGreaterThanZero / totalDocuments) * 100;
      const percentageXGreaterThanZeroAndYLessThanZero = (xGreaterThanZeroAndYLessThanZero / totalDocuments) * 100;
      const percentageXLessThanZeroAndYLessThanZero = (xLessThanZeroAndYLessThanZero / totalDocuments) * 100;

      // Add labels to display percentages in the squares
      // const squareSize = 300; // Adjust this based on your map's layout
      // const labelOffsetX = 10; // Adjust these offsets as needed
      // const labelOffsetY = 20;

      // Calculate the maximum X and Y coordinates

      // Calculate the midpoints for the squares
      const xMid = d3.max(docsData, (d) => d.x) / 2;
      const yMid = d3.max(docsData, (d) => d.y) / 2;

      // Labels for X > 0 and Y > 0 square
      svg
        .append("text")
        .attr("x", xScale(xMid))
        .attr("y", yScale(yMid))
        .text(`${percentageXGreaterThanZeroAndYGreaterThanZero.toFixed(0)}%`) // Remove the prefix
        .style("text-anchor", "middle")
        .style("fill", "dark") // Change the text color to blue
        .style("font-size", "100px") // Adjust the font size
        .style("opacity", 0.1); // Adjust the opacity (0.7 means slightly transparent)

      // Labels for X < 0 and Y > 0 square
      svg
        .append("text")
        .attr("x", xScale(-xMid))
        .attr("y", yScale(yMid))
        .text(`${percentageXLessThanZeroAndYGreaterThanZero.toFixed(0)}%`) // Remove the prefix
        .style("text-anchor", "middle")
        .style("fill", "dark") // Change the text color to light blue
        .style("font-size", "100px") // Adjust the font size
        .style("opacity", 0.1); // Adjust the opacity (0.05 means slightly transparent)

      // Labels for X > 0 and Y < 0 square
      svg
        .append("text")
        .attr("x", xScale(xMid))
        .attr("y", yScale(-yMid))
        .text(`${percentageXGreaterThanZeroAndYLessThanZero.toFixed(0)}%`) // Remove the prefix
        .style("text-anchor", "middle")
        .style("fill", "dark") // Change the text color to light blue
        .style("font-size", "100px") // Adjust the font size
        .style("opacity", 0.1); // Adjust the opacity (0.05 means slightly transparent)

      // Labels for X > 0 and Y < 0 square
      svg
        .append("text")
        .attr("x", xScale(-xMid))
        .attr("y", yScale(-yMid))
        .text(`${percentageXLessThanZeroAndYLessThanZero.toFixed(0)}%`) // Remove the prefix
        .style("text-anchor", "middle")
        .style("fill", "dark") // Change the text color to light blue
        .style("font-size", "100px") // Adjust the font size
        .style("opacity", 0.1); // Adjust the opacity (0.05 means slightly transparent)

      const topicsPolygons = svg
        .selectAll("polygon.topic-polygon")
        .data(topicsCentroids)
        .enter()
        .append("polygon")
        .attr("class", "topic-polygon")
        .attr("points", (d) => {
          const hull = d.convex_hull;
          if (hull) {
            const hullPoints = hull.x_coordinates.map((x, i) => [xScale(x), yScale(hull.y_coordinates[i])]);
            return hullPoints.map((point) => point.join(",")).join(" ");
          }
        })
        .style("fill", "transparent")
        .style("stroke", "transparent")
        .style("stroke-width", 2);

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

        if (d.top_doc_content) {
          const topicName = d.name;
          const topicSize = d.size;
          const totalSize = topicsCentroids.reduce((sum, topic) => sum + topic.size, 0);
          const sizeFraction = Math.round((topicSize / totalSize) * 100);
          const content = d.top_doc_content;

          ReactDOM.render(<TextContainer topicName={topicName} sizeFraction={sizeFraction} content={content} />, textContainerRef.current);
        } else {
          textContainerRef.current.innerHTML = "No content available for this topic.";
        }
      });
    },
    [svgHeight, svgWidth],
  );

  useEffect(() => {
    if (REACT_APP_API_ENDPOINT === "local" || apiData === undefined) {
      // Fetch the JSON data locally
      fetch(`/${bunkaDocs}`)
        .then((response) => response.json())
        .then((docsData) => {
          // Fetch the local topics data and merge it with the existing data
          fetch(`/${bunkaTopics}`)
            .then((response) => response.json())
            .then((topicsData) => {
              fetch(`/${bunkaQuery}`)
                .then((response) => response.json())
                .then((queryData) => {
                  // Call the function to create the scatter plot after data is loaded
                  createScatterPlot(docsData, topicsData, queryData);
                })
                .catch((error) => {
                  console.error("Error fetching bourdieu query data:", error);
                });
            })
            .catch((error) => {
              console.error("Error fetching topics data:", error);
            });
        })
        .catch((error) => {
          console.error("Error fetching documents data:", error);
        });
    } else {
      console.log(apiData);
      // Call the function to create the scatter plot with the data provided by TopicsContext
      createScatterPlot(apiData.docs, apiData.topics, apiData.query);
    }
  }, [apiData, createScatterPlot]);
   
  const mapDescription = "This map is generated by projecting documents onto a two-dimensional space, where the axes are defined by the user. Two documents are positioned close to each other if they share a similar relationship with the axes. The documents themselves are not directly represented on the map; rather, they are aggregated into clusters. Each cluster represents a group of documents that exhibit similarities.";
   
  return (
    <div className="json-display">
      {isLoading ? (
        <Backdrop open={isLoading} style={{ zIndex: 9999 }}>
          <CircularProgress color="primary" />
        </Backdrop>
      ) : (
        <div className="scatter-plot-and-text-container">
          <div className="scatter-plot-container" ref={scatterPlotContainerRef}>
            <Tooltip
              style={{ position: "relative", top: 10, left: 40 }}
              title={mapDescription}
            >
              <HelpIcon />
            </Tooltip>
            <svg ref={svgRef} />
          </div>
          <div className="text-container" ref={textContainerRef}>
            {selectedDocument ? (
              <>
                <Box marginBottom={2}>
                  <Button component="label" variant="outlined" startIcon={<RepeatIcon />} onClick={setSelectedDocument(null)}>
                    Upload another CSV file
                  </Button>
                </Box>
                <div className="text-content">
                  <h2 className="topic-name">
                    Topic:
                    {selectedDocument.topic_id}
                  </h2>

                  <p>{selectedDocument.content}</p>
                </div>
              </>
            ): <QueryView />}
          </div>
        </div>
      )}
    </div>
  );
}

export default Bourdieu;
