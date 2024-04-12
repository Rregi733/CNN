using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace CNN.Model
{
    public class DataSerializer
    {
        public void BinarySerialize(object data, string filePath)
        {
            FileStream fs;
            BinaryFormatter bf = new BinaryFormatter();
            if (File.Exists(filePath)) File.Delete(filePath);
            fs = File.Create(filePath);
            bf.Serialize(fs, data);
            fs.Close();
        }

        public object BinaryDeserialize(string filePath)
        {
            object obj = null;

            FileStream fs;
            BinaryFormatter bf = new BinaryFormatter();
            if (File.Exists(filePath))
            {
                fs = File.OpenRead(filePath);
                obj = bf.Deserialize(fs);
                fs.Close();
            }

            return obj;
        }
    }
}
