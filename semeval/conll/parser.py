# Zephyr - parses their awful conll format to something actually usable...
def parse_conll_to_json(inputPath,outputPath):
    with open(input) as fp:
        output =[]
        count= 0
        instance={}
        instance['tokens']=[]
        instance['langid']=[]
        instance['tweet']=""
        for line in fp.readlines():
            if re.search(r'^meta\b\t[0-9]',line):
                meta = line.replace('\n','').split('\t')
                # print(meta)
                instance['tweetid'] = int(meta[1].strip())
                instance['sentiment'] = meta[2].strip()
            elif line == '\n':
                count+=1
                output.append(instance)
                instance={}
                instance['tokens']=[]
                instance['langid']=[]
                instance['tweet']=""
                # print('found boundary')
            # print(line)
            else:
                parts = line.replace('\n','').split('\t')
                # print(parts)
                instance['tweet'] = "%s %s" % (instance['tweet'], parts[0])
                instance['tokens'].append(parts[0])
                instance['langid'].append(parts[1])
        print(output)
        with open(outputPath, 'w') as fp:
            json.dump(output, fp)
