package com.project;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardWatchEventKinds;
import java.nio.file.WatchEvent;
import java.nio.file.WatchKey;
import java.nio.file.WatchService;
import java.util.concurrent.CompletableFuture;

public abstract class ObservableFile {
    
    String path;
    String name;
    CompletableFuture<Void> future = null;

    public ObservableFile (File file) {
        this.path = (Paths.get(file.getAbsolutePath())).getParent().toString();
        this.name = file.getName();
        
        if (this.path.charAt(this.path.length() - 1) == '.') {
            this.path = this.path.substring(0, this.path.length() - 1);
        }

        future = CompletableFuture.runAsync(getRunnable(this));
    }

    public abstract void onChange();

    static Runnable getRunnable (ObservableFile obj) {
        return new Runnable () {
            @Override
            public void run () {
                boolean running = true;
                try {
                    WatchService watcher = FileSystems.getDefault().newWatchService();
                    Path dir = Paths.get(obj.path);
                    dir.register(watcher, 
                        StandardWatchEventKinds.ENTRY_CREATE, 
                        StandardWatchEventKinds.ENTRY_DELETE, 
                        StandardWatchEventKinds.ENTRY_MODIFY);
                     
                    System.out.printf("Thread vigilant l'arxiu \"%s\" del path \"%s\"\n", obj.name, dir);
                     
                    while (running) {
                        WatchKey key;
                        try {
                            key = watcher.take();
                        } catch (InterruptedException ex) {
                            return;
                        }
                         
                        for (WatchEvent<?> event : key.pollEvents()) {
                            WatchEvent.Kind<?> kind = event.kind();
                             
                            @SuppressWarnings("unchecked")
                            WatchEvent<Path> ev = (WatchEvent<Path>) event;
                            Path fileName = ev.context();

                            if (kind == StandardWatchEventKinds.ENTRY_MODIFY
                             && fileName.toString().equals(obj.name)) {
                                obj.onChange();
                            }
                        }
                         
                        boolean valid = key.reset();
                        if (!valid) {
                            break;
                        }
                    }
                     
                } catch (IOException ex) {
                    System.err.println(ex);
                }
            }
        };
    }
}