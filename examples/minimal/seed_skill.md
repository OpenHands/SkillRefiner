---
name: xlsx
description: "This skill should be used when the user asks to edit, compute over, or repair an Excel spreadsheet: filling a column, writing a formula, reshaping a sheet, or saving a modified workbook to a required output path."
---

# Spreadsheet Manipulation

Solve one spreadsheet task by executing Python against the workbook, then saving
the result to the exact output path the task names.

## Workflow

1. **Read the task's header fields first.** `spreadsheet_path`, `output_path`,
   `answer_position` and `instruction_type` define the contract. The answer must
   land in `answer_position` of the file at `output_path`.
2. **Inspect before editing.** Load the workbook, print the sheet names and the
   used range, and look at the first rows so column letters are resolved from the
   data rather than assumed.
3. **Edit with openpyxl.** Prefer `openpyxl.load_workbook(path)` and write cells
   directly. Keep the original sheet order and names unless the task says
   otherwise.
4. **Save to `output_path` exactly.** Do not save in place, and do not invent a
   filename.
5. **Verify.** Reload the saved workbook and print the cells you wrote before
   finishing.

## Notes

- Work only inside the working directory the task names.
- Preserve existing formatting and untouched cells; write the answer range only.
- When the task asks for a formula, write the formula string, not its computed
  value.
