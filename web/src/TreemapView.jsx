import { Backdrop, CircularProgress, List, ListItem, Paper, Typography } from "@mui/material";
import * as d3 from "d3";
import React, { useCallback, useContext, useEffect, useRef, useState } from "react";
import { TopicsContext } from "./UploadFileContext";

const bunkaTopics = "bunka_topics.json";
const { REACT_APP_API_ENDPOINT } = process.env;

function TreemapView() {
  const svgRef = useRef(null);
  const [selectedTopic, setSelectedTopic] = useState({ name: "", content: [] });
  const { data: apiData, isLoading } = useContext(TopicsContext);

  const createTreemap = useCallback((data) => {
    const width = window.innerWidth * 0.6; // Adjust the width for the treemap
    const height = 800; // Adjust the height as needed

    const svg = d3.select(svgRef.current).attr("width", width).attr("height", height);

    const root = d3.hierarchy({ children: data }).sum((d) => d.size);

    const treemapLayout = d3.treemap().size([width, height]).padding(1).round(true);

    treemapLayout(root);

    const cell = svg
      .selectAll("g")
      .data(root.leaves())
      .enter()
      .append("g")
      .attr("transform", (d) => `translate(${d.x0},${d.y0})`)
      .on("click", (event, d) => {
        const topicName = d.data.name;
        const topicContent = d.data.top_doc_content || [];

        setSelectedTopic({ name: topicName, content: topicContent });
      });

    cell
      .append("rect")
      .attr("width", (d) => d.x1 - d.x0)
      .attr("height", (d) => d.y1 - d.y0)
      .style("fill", "lightblue")
      .style("stroke", "blue");

    cell
      .append("text")
      .selectAll("tspan")
      .data((d) => {
        const text = d.data.name.split(/(?=[A-Z][^A-Z])/g); // Split topic name on capital letters
        return text;
      })
      .enter()
      .append("tspan")
      .attr("x", 3)
      .attr("y", (d, i) => 13 + i * 10)
      .text((d) => d);

    svg.selectAll("text").attr("font-size", 13).attr("fill", "black");
  }, []);

  useEffect(() => {
    if (REACT_APP_API_ENDPOINT === "local" || apiData === undefined) {
      // Fetch the JSON data locally
      fetch(`/${bunkaTopics}`)
        .then((response) => response.json())
        .then((localData) => {
          createTreemap(localData);
        })
        .catch((error) => {
          console.error("Error fetching JSON data:", error);
        });
    } else {
      // Call the function with the data provided by TopicsContext
      createTreemap(apiData.topics);
    }
  }, [apiData, createTreemap]);

  return (
    <div>
      <h2>Treemap View</h2>
      {isLoading ? (
        <Backdrop open={isLoading} style={{ zIndex: 9999 }}>
          <CircularProgress color="primary" />
        </Backdrop>
      ) : (
        <div style={{ display: "flex" }}>
          <svg ref={svgRef} style={{ marginRight: "20px" }} />
          <div
            style={{
              width: window.innerWidth * 0.25,
              maxHeight: "800px",
              overflowY: "auto",
            }}
          >
            <Paper>
              <Typography
                variant="h4"
                style={{
                  position: "sticky",
                  top: 0,
                  backgroundColor: "white",
                  color: "blue",
                }}
              >
                {selectedTopic.name}
              </Typography>
              {selectedTopic.content.map((doc, index) => (
                <List key={doc.id}>
                  <ListItem>
                    <Typography variant="h5">{doc}</Typography>
                  </ListItem>
                </List>
              ))}
              {selectedTopic.content.length === 0 && <Typography variant="h4">Click on a Square.</Typography>}
            </Paper>
          </div>
        </div>
      )}
    </div>
  );
}

export default TreemapView;
