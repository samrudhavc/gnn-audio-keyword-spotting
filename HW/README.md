# Hardware module

To work conveniently with the GitHub repository and Vivado software, you need to: 
1. Create Vivado project.
2. Add Sources from `HW/src` dicrectory (`.sv` files for design and `.xci` files for IPs). **Remeber to deselect "Copy sources intro project"**.
3. Add constraints, select the target `Part` or `Board`, Finish!

In this way all changes to already created files will be recorded. **In order to create new desing/simulation source** you should first create new file in repository and than add it to Vivado project (`Add Sources Alt+A`) without Coping them. It would be most convenient to work on separate branches and meet before merging. 