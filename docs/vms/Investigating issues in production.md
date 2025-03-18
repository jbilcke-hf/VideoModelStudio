# Investigating issue in production

Note: the following document is meant for maintainers of VMS.

It describe things that are normally only useful during development, for instance if there are bugs.

Normal VMS users do not need to read this document and perform those steps, since in theory VMS is already taking care of things (eg. automatic fix of corrupted internal JSON files) or providing ways to solve common issues (eg. buttons to download or delete data).

## Backuping data

During development of VMs, there might be bugs creating data corruption.

To avoid reseting the /data folder all the time, I suggest simply doing backups of what you need.

1. run the space in developer mode
2. open vscode from the dev mode panel
3. go to the data dir (eg /data if you have persistence, or .data otherwise)
4. do whatever you want (you are the developer, after all)
5. for instance you can edit files, delete stuff etc

### Manual backout of the output dir

```bash
mkdir /data/backup

# to copy training data
cp -r /data/training /data/backup/training

# to copy generated models and checkpoints
cp -r /data/output /data/backup/output
```

### Manual restore of a backup

```bash
# if you already have a backup of output/ then you can delete its content
rm -Rf output/*

# restore the backup, for instance the weights and checkpoints
cp -r backup/output/* output/
```

### Manual restore of UI state

Restoring the UI state can be tricky as it is being modified by Gradio.

I recommend shutting Gradio down, but this will kill the space and the VS Code session.

So a tricky is to restart Gradio and immediately perform this command:

```bash
cp backup/output/ui_state.json output/
```

That way Gradio will inialize itself with the backuped UI state.