/**
 * Hanlde SSE for progressing tasks progress tracking
 * @param {String} task_id
 * @param {function} callback
 * @returns
 */
function sseTaskQuery(task_id, callback) {
  const evtSource = new EventSource(`/tasks/${task_id}/progress`);
  evtSource.onmessage = function (event) {
    const data = JSON.parse(event.data);
    console.log("Task Progress:", data);
    if (callback) {
      callback(data);
    }
    if (data.state === "SUCCESS" || data.state === "FAILURE") {
      evtSource.close(); // Close the connection when the task is complete or failed
    }
  };
  return evtSource;
}

export default sseTaskQuery;
